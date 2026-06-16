import json
from pathlib import Path
import tempfile
import unittest

from activity_patterns.events import iter_zeek_events, record_to_event
from activity_patterns.prepare import contiguous_chunks, write_sequences


class RecordToEventTests(unittest.TestCase):
    def test_conn_event_excludes_aggregate_and_identifier_fields(self):
        event = record_to_event(
            "conn",
            {
                "ts": 2.0,
                "uid": "leaky-id",
                "id.orig_h": "10.0.0.1",
                "id.orig_p": 49152,
                "id.resp_p": 80,
                "proto": "tcp",
                "service": "http",
                "conn_state": "S0",
                "history": "S",
                "duration": 100.0,
                "orig_bytes": 999999,
                "orig_pkts": 500,
            },
        )

        self.assertEqual(
            event.tokens,
            (
                "log=conn",
                "proto=tcp",
                "service=http",
                "conn_state=S0",
                "history=S",
                "resp_port=80",
            ),
        )
        flattened = " ".join(event.tokens)
        self.assertNotIn("10.0.0.1", flattened)
        self.assertNotIn("duration", flattened)
        self.assertNotIn("orig_bytes", flattened)

    def test_application_protocol_fields_are_direct_tokens(self):
        event = record_to_event(
            "dns",
            {
                "ts": 1.0,
                "qtype_name": "A",
                "rcode_name": "NOERROR",
                "opcode_name": "query",
                "rejected": False,
            },
        )
        self.assertEqual(
            event.tokens,
            (
                "log=dns",
                "qtype_name=A",
                "rcode_name=NOERROR",
                "opcode_name=query",
                "rejected=false",
            ),
        )


class SequencePreparationTests(unittest.TestCase):
    def test_logs_are_merged_in_timestamp_order_and_chunked_without_overlap(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "conn.log").write_text(
                json.dumps({"ts": 3.0, "proto": "tcp", "conn_state": "SF"}) + "\n"
                + json.dumps({"ts": 1.0, "proto": "tcp", "conn_state": "S0"})
                + "\n",
                encoding="utf-8",
            )
            (root / "dns.log").write_text(
                json.dumps({"ts": 2.0, "qtype_name": "A"}) + "\n",
                encoding="utf-8",
            )

            events = list(iter_zeek_events(root))
            self.assertEqual([event.timestamp for event in events], [1.0, 2.0, 3.0])

            chunks = list(contiguous_chunks(events, max_events=2))
            self.assertEqual(
                [[event.timestamp for event in chunk] for chunk in chunks],
                [[1.0, 2.0], [3.0]],
            )

    def test_writer_keeps_capture_id_for_grouped_splitting(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            logs = root / "logs"
            logs.mkdir()
            (logs / "conn.log").write_text(
                json.dumps({"ts": 1.0, "proto": "tcp"}) + "\n",
                encoding="utf-8",
            )
            output = root / "sequence.jsonl"

            count = write_sequences(
                logs,
                output,
                label="Recon",
                sequence_id="capture-001",
                max_events=10,
            )

            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(count, 1)
            self.assertEqual(payload["sequence_id"], "capture-001")
            self.assertEqual(payload["label"], "Recon")


if __name__ == "__main__":
    unittest.main()

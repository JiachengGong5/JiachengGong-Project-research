import json
from pathlib import Path
import tempfile
import unittest

from activity_patterns.trace import (
    reconstruct_activity_trace,
    resolve_zeek_log_dir,
)


class TraceReconstructionTests(unittest.TestCase):
    def test_sample_sequence_path_resolves_to_sample_zeek_directory(self):
        path = Path("data/sequences/Mirai-greip_flood/Mirai-greip_flood.sample.jsonl")

        resolved = resolve_zeek_log_dir(path, project_root="/project")

        self.assertEqual(
            resolved,
            Path("/project/data/zeek_sample/Mirai-greip_flood/Mirai-greip_flood"),
        )

    def test_trace_reconstruction_includes_uid_linked_protocol_records(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            conn_record = {
                "ts": 10.0,
                "uid": "C1",
                "id.orig_h": "10.0.0.2",
                "id.resp_h": "10.0.0.3",
                "id.resp_p": 80,
                "proto": "tcp",
                "service": "http",
                "conn_state": "SF",
            }
            http_record = {
                "ts": 12.0,
                "uid": "C1",
                "id.orig_h": "10.0.0.2",
                "id.resp_h": "10.0.0.3",
                "method": "GET",
                "uri": "/index.html",
                "status_code": 200,
            }
            unrelated_record = {
                "ts": 10.2,
                "uid": "C2",
                "id.orig_h": "10.0.0.4",
                "id.resp_h": "10.0.0.5",
                "proto": "udp",
            }
            (root / "conn.log").write_text(
                json.dumps(conn_record) + "\n" + json.dumps(unrelated_record) + "\n",
                encoding="utf-8",
            )
            (root / "http.log").write_text(
                json.dumps(http_record) + "\n",
                encoding="utf-8",
            )

            trace = reconstruct_activity_trace(
                root,
                start_ts=10.0,
                end_ts=10.0,
                context_seconds=0.0,
                max_records=10,
            )

            self.assertEqual([record["log_type"] for record in trace], ["conn", "http"])
            self.assertEqual(trace[1]["uri"], "/index.html")


if __name__ == "__main__":
    unittest.main()

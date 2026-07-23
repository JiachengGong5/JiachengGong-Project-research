import json
from pathlib import Path
import tempfile
import unittest

from activity_patterns.coverage import FINE_LABEL_TO_FOLDER
from activity_patterns.labels import (
    CATEGORY_TO_FINE_LABELS,
    FINE_LABEL_NAMES,
    canonical_fine_label,
    category_index,
    coarse_label_for_fine,
    fine_label_from_category_index,
    fine_index_within_category,
)
from activity_patterns.manifest import (
    ManifestRow,
    build_manifest_rows,
    read_manifest,
    write_manifest,
)
from activity_patterns.vocab import PAD_TOKEN, UNK_TOKEN, build_vocab
from activity_patterns.rechunk import rechunk_manifest_rows

try:
    import torch
    from torch.utils.data import DataLoader

    from activity_patterns.dataset import SequenceChunkDataset, collate_sequence_chunks
    from activity_patterns.labels import CATEGORY_NAMES, CATEGORY_TO_NUM_FINE
    from activity_patterns.model import HierarchicalProtocolEventLSTM, hierarchical_loss
except ImportError:  # pragma: no cover - optional model dependency
    torch = None


def write_sequence(path: Path, sequence_id: str, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payloads = [
        {
            "sequence_id": sequence_id,
            "chunk_index": 0,
            "label": label,
            "events": [
                {
                    "timestamp": 1.0,
                    "log_type": "conn",
                    "tokens": ["log=conn", "proto=tcp", "conn_state=S0"],
                },
                {
                    "timestamp": 2.0,
                    "log_type": "dns",
                    "tokens": ["log=dns", "qtype_name=A"],
                },
            ],
        },
        {
            "sequence_id": sequence_id,
            "chunk_index": 1,
            "label": label,
            "events": [
                {
                    "timestamp": 3.0,
                    "log_type": "conn",
                    "tokens": ["log=conn", "proto=udp", "conn_state=SF"],
                }
            ],
        },
    ]
    with path.open("w", encoding="utf-8") as handle:
        for payload in payloads:
            handle.write(json.dumps(payload) + "\n")


class LabelTests(unittest.TestCase):
    def test_dataset_folder_labels_normalize_to_canonical_hierarchy(self):
        self.assertEqual(canonical_fine_label("Benign_Final"), "Benign")
        self.assertEqual(canonical_fine_label("DDoS-SYN_Flood"), "DDoS-SYN flood")
        self.assertEqual(
            canonical_fine_label("DDoS-ICMP_Fragmentation"),
            "DDoS-ICMP fragmentation",
        )
        self.assertEqual(canonical_fine_label("Recon-PortScan"), "Recon-Port scan")
        self.assertEqual(coarse_label_for_fine("Recon-PortScan"), "Recon")
        self.assertEqual(category_index("Recon"), 3)
        self.assertEqual(fine_index_within_category("Recon-Port scan"), 3)
        self.assertEqual(
            fine_label_from_category_index("Recon", 3),
            "Recon-Port scan",
        )

    def test_all_fine_labels_have_download_folder_names(self):
        self.assertEqual(len(FINE_LABEL_NAMES), 34)
        self.assertEqual(len(CATEGORY_TO_FINE_LABELS["DDoS"]), 12)
        self.assertEqual(set(FINE_LABEL_TO_FOLDER), set(FINE_LABEL_NAMES))
        self.assertEqual(
            canonical_fine_label(FINE_LABEL_TO_FOLDER["DDoS-ACK fragmentation"]),
            "DDoS-ACK fragmentation",
        )
        self.assertEqual(
            canonical_fine_label(FINE_LABEL_TO_FOLDER["DoS-UDP flood"]),
            "DoS-UDP flood",
        )


class ManifestAndVocabTests(unittest.TestCase):
    def test_rechunk_preserves_events_labels_and_parent_splits(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_path = (
                root
                / "data"
                / "sequences"
                / "Benign_Final"
                / "BenignTraffic.jsonl"
            )
            write_sequence(source_path, "BenignTraffic", "Benign")
            rows = [
                ManifestRow(
                    sequence_path=source_path,
                    sequence_id="BenignTraffic:chunk-000000",
                    coarse_label="Benign",
                    fine_label="Benign",
                    split="train",
                    chunk_index=0,
                ),
                ManifestRow(
                    sequence_path=source_path,
                    sequence_id="BenignTraffic:chunk-000001",
                    coarse_label="Benign",
                    fine_label="Benign",
                    split="val",
                    chunk_index=1,
                ),
            ]

            output_rows, stats = rechunk_manifest_rows(
                rows,
                output_root=root / "data" / "sequences_windowed",
                project_root=root,
                max_events=1,
            )

            self.assertEqual(stats["input_events"], 3)
            self.assertEqual(stats["output_events"], 3)
            self.assertEqual(stats["output_chunks"], 3)
            self.assertEqual(
                [row.split for row in output_rows],
                ["train", "train", "val"],
            )
            output_path = output_rows[0].sequence_path
            payloads = [
                json.loads(line)
                for line in output_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual([len(payload["events"]) for payload in payloads], [1, 1, 1])
            self.assertEqual(
                [payload["source_chunk_index"] for payload in payloads],
                [0, 0, 1],
            )
            self.assertEqual(
                [(payload["event_start"], payload["event_stop"]) for payload in payloads],
                [(0, 1), (1, 2), (0, 1)],
            )

    def test_manifest_and_vocab_are_built_from_sequence_files(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            sequence_root = root / "data" / "sequences"
            write_sequence(
                sequence_root / "Benign_Final" / "BenignTraffic.jsonl",
                "BenignTraffic",
                "Benign",
            )
            write_sequence(
                sequence_root / "Recon-PortScan" / "Recon-PortScan.jsonl",
                "Recon-PortScan",
                "Recon-PortScan",
            )

            rows = build_manifest_rows(sequence_root, project_root=root)
            manifest_path = root / "manifests" / "smoke.csv"
            write_manifest(rows, manifest_path)
            loaded_rows = read_manifest(manifest_path, project_root=root)
            vocab = build_vocab(loaded_rows)

            self.assertEqual(len(loaded_rows), 2)
            self.assertEqual(loaded_rows[0].coarse_label, "Benign")
            self.assertEqual(loaded_rows[1].fine_label, "Recon-Port scan")
            self.assertEqual(vocab.token_to_id[PAD_TOKEN], 0)
            self.assertEqual(vocab.token_to_id[UNK_TOKEN], 1)
            self.assertIn("conn_state=S0", vocab.token_to_id)

    def test_chunk_level_manifest_rows_select_only_requested_chunks(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            sequence_path = root / "data" / "sequences" / "Benign_Final" / "BenignTraffic.jsonl"
            write_sequence(sequence_path, "BenignTraffic", "Benign")

            relative_path = Path("data/sequences/Benign_Final/BenignTraffic.jsonl")
            rows = [
                ManifestRow(
                    sequence_path=relative_path,
                    sequence_id="BenignTraffic:chunk-000001",
                    coarse_label="Benign",
                    fine_label="Benign",
                    split="train",
                    chunk_index=1,
                ),
                ManifestRow(
                    sequence_path=relative_path,
                    sequence_id="BenignTraffic:chunk-000000",
                    coarse_label="Benign",
                    fine_label="Benign",
                    split="val",
                    chunk_index=0,
                ),
            ]
            manifest_path = root / "manifests" / "dev.csv"
            write_manifest(rows, manifest_path)
            loaded_rows = read_manifest(manifest_path, project_root=root)
            vocab = build_vocab([loaded_rows[0]])

            train_dataset = SequenceChunkDataset(
                loaded_rows,
                vocab,
                split="train",
                max_events=8,
            )
            val_dataset = SequenceChunkDataset(
                loaded_rows,
                vocab,
                split="val",
                max_events=8,
            )

            self.assertEqual(loaded_rows[0].chunk_index, 1)
            self.assertEqual(len(train_dataset), 1)
            self.assertEqual(len(val_dataset), 1)
            self.assertEqual(train_dataset[0]["chunk_index"], 1)
            self.assertEqual(val_dataset[0]["chunk_index"], 0)
            self.assertIn("conn_state=SF", vocab.token_to_id)
            self.assertNotIn("conn_state=S0", vocab.token_to_id)


@unittest.skipIf(torch is None, "PyTorch is not installed")
class DatasetTests(unittest.TestCase):
    def test_dataset_collates_chunks_for_hierarchical_model(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            sequence_root = root / "data" / "sequences"
            write_sequence(
                sequence_root / "Benign_Final" / "BenignTraffic.jsonl",
                "BenignTraffic",
                "Benign",
            )
            write_sequence(
                sequence_root / "DDoS-SYN_Flood" / "DDoS-SYN_Flood.sample.jsonl",
                "DDoS-SYN_Flood-sample",
                "DDoS-SYN_Flood",
            )

            rows = build_manifest_rows(sequence_root, project_root=root)
            manifest_path = root / "manifests" / "smoke.csv"
            write_manifest(rows, manifest_path)
            loaded_rows = read_manifest(manifest_path, project_root=root)
            vocab = build_vocab(loaded_rows)

            dataset = SequenceChunkDataset(loaded_rows, vocab, max_events=8)
            loader = DataLoader(
                dataset,
                batch_size=2,
                collate_fn=collate_sequence_chunks,
            )
            batch = next(iter(loader))

            self.assertEqual(tuple(batch["token_ids"].shape[:2]), (2, 2))
            self.assertEqual(tuple(batch["lengths"].shape), (2,))
            self.assertEqual(tuple(batch["coarse_targets"].shape), (2,))

            model = HierarchicalProtocolEventLSTM(
                len(vocab),
                CATEGORY_TO_NUM_FINE,
                embedding_dim=8,
                hidden_dim=12,
                num_layers=1,
            )
            outputs = model(batch["token_ids"], batch["lengths"])
            loss = hierarchical_loss(
                outputs,
                batch["coarse_targets"],
                batch["fine_targets"],
                CATEGORY_NAMES,
            )

            self.assertEqual(tuple(outputs["coarse"].shape), (2, len(CATEGORY_NAMES)))
            self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()

import json
from pathlib import Path
import sys
import tempfile
import unittest

from activity_patterns.manifest import ManifestRow, build_manifest_rows
from activity_patterns.labels import CICIOT2023_SCHEMA
from activity_patterns.schema import LabelSchema, label_schema_from_checkpoint
from activity_patterns.vocab import build_vocab

try:
    import torch

    from activity_patterns.dataset import SequenceChunkDataset
    from activity_patterns.model import HierarchicalProtocolEventLSTM
except ImportError:  # pragma: no cover - optional model dependency
    torch = None


SCRIPTS_ROOT = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_ROOT))

from build_dev_split_manifest import assign_development_splits  # noqa: E402


def write_sequence(path: Path, label: str, sequence_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "sequence_id": sequence_id,
        "chunk_index": 0,
        "label": label,
        "events": [
            {
                "timestamp": 1.0,
                "log_type": "conn",
                "tokens": ["log=conn", "proto=tcp", f"service={label}"],
            }
        ],
    }
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


class LabelSchemaTests(unittest.TestCase):
    def test_flat_schema_is_inferred_and_round_trips(self):
        schema = LabelSchema.from_flat_labels(
            ["Normal", "PortScan"],
            name="portable-test",
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "schema.json"
            schema.save(path)
            loaded = LabelSchema.load(path)

        self.assertEqual(loaded.category_names, ("Normal", "PortScan"))
        self.assertEqual(loaded.fine_label_names, ("Normal", "PortScan"))
        self.assertEqual(loaded.coarse_label_for_fine("port_scan"), "PortScan")

    def test_hierarchical_aliases_drive_manifest_labels(self):
        schema = LabelSchema(
            name="custom",
            category_to_fine_labels={
                "Normal": ("Benign",),
                "Recon": ("Port Scan", "Host Discovery"),
            },
            aliases={"normal_traffic": "Benign", "port_scan": "Port Scan"},
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            sequence_root = root / "sequences"
            write_sequence(
                sequence_root / "normal_traffic" / "capture-a.jsonl",
                "Benign",
                "capture-a",
            )
            write_sequence(
                sequence_root / "port_scan" / "capture-b.jsonl",
                "Port Scan",
                "capture-b",
            )

            rows = build_manifest_rows(
                sequence_root,
                project_root=root,
                label_schema=schema,
            )

        self.assertEqual(
            [(row.coarse_label, row.fine_label) for row in rows],
            [("Normal", "Benign"), ("Recon", "Port Scan")],
        )

    def test_auto_split_keeps_multi_capture_classes_disjoint(self):
        rows = [
            ManifestRow(
                sequence_path=Path(f"capture-{capture}.jsonl"),
                sequence_id=f"{capture}-{chunk}",
                coarse_label="A",
                fine_label="A",
                chunk_index=chunk,
            )
            for capture in ("one", "two")
            for chunk in range(2)
        ]

        split_rows = assign_development_splits(
            rows,
            val_ratio=0.5,
            stratify_by="fine",
            seed=7,
            split_unit="auto",
        )
        splits_by_capture = {}
        for row in split_rows:
            splits_by_capture.setdefault(row.sequence_path, set()).add(row.split)

        self.assertEqual(
            {frozenset(splits) for splits in splits_by_capture.values()},
            {frozenset({"train"}), frozenset({"val"})},
        )

    def test_legacy_checkpoint_head_counts_restrict_the_fallback_schema(self):
        schema = label_schema_from_checkpoint(
            {
                "category_names": CICIOT2023_SCHEMA.category_names,
                "category_to_num_fine": {
                    **CICIOT2023_SCHEMA.category_to_num_fine,
                    "DDoS": 11,
                },
            },
            fallback=CICIOT2023_SCHEMA,
        )

        self.assertEqual(len(schema.category_to_fine_labels["DDoS"]), 11)
        self.assertNotIn(
            "DDoS-ICMP fragmentation",
            schema.category_to_fine_labels["DDoS"],
        )


@unittest.skipIf(torch is None, "PyTorch is not installed")
class PortableDatasetModelTests(unittest.TestCase):
    def test_custom_schema_controls_dataset_targets_and_model_heads(self):
        schema = LabelSchema(
            name="custom-model",
            category_to_fine_labels={
                "Normal": ("Benign",),
                "Attack.v1": ("SYN", "UDP"),
            },
            aliases={"syn_capture": "SYN"},
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            sequence_path = root / "syn_capture" / "capture.jsonl"
            write_sequence(sequence_path, "SYN", "capture")
            rows = build_manifest_rows(
                root,
                project_root=root / "unrelated-project-root",
                label_schema=schema,
            )
            vocab = build_vocab(rows)
            dataset = SequenceChunkDataset(
                rows,
                vocab,
                label_schema=schema,
            )
            item = dataset[0]

            model = HierarchicalProtocolEventLSTM(
                len(vocab),
                schema.category_to_num_fine,
                embedding_dim=4,
                hidden_dim=6,
                num_layers=1,
            )
            token_ids = torch.tensor([[item["token_ids"][0]]])
            lengths = torch.tensor([1])
            outputs = model(token_ids, lengths)

        self.assertEqual(item["coarse_target"], 1)
        self.assertEqual(item["fine_target"], 0)
        self.assertEqual(tuple(outputs["coarse"].shape), (1, 2))
        self.assertEqual(tuple(outputs["fine"]["Attack.v1"].shape), (1, 2))


if __name__ == "__main__":
    unittest.main()

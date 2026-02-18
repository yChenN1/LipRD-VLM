import csv
import json
from pathlib import Path
from typing import Any, Dict, List

from torch.utils.data import Dataset


def _read_json(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"JSON must be a list of samples: {path}")
    return data


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _read_csv_or_tsv(path: Path, delimiter: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        for row in reader:
            rows.append(dict(row))
    return rows


class LipReadingDataset(Dataset):
    """
    Expected fields for each row:
      - `video` or `video_path`: path to input video.
      - `text` or `label`: ground-truth transcription for lip reading.
      - `instruction` (optional): per-sample instruction.
    """

    def __init__(
        self,
        ann_path: str = "",
        data_root: str = "",
        default_instruction: str = "Please read the speaker's lips and transcribe the speech.",
    ) -> None:
        self.ann_path = Path(ann_path) if ann_path else None
        self.data_root = Path(data_root) if data_root else None
        self.default_instruction = default_instruction
        self.samples = self._build_samples()
        if not self.samples:
            source = str(self.data_root) if self.data_root is not None else str(self.ann_path)
            raise ValueError(f"No valid sample found in: {source}")

    def _build_samples(self) -> List[Dict[str, Any]]:
        if self.data_root is not None:
            return self._load_from_root(self.data_root)
        if self.ann_path is not None:
            return self._load_samples(self.ann_path)
        raise ValueError("Either ann_path or data_root must be provided.")

    @staticmethod
    def _load_samples(path: Path) -> List[Dict[str, Any]]:
        suffix = path.suffix.lower()
        if suffix == ".json":
            return _read_json(path)
        if suffix == ".jsonl":
            return _read_jsonl(path)
        if suffix == ".csv":
            return _read_csv_or_tsv(path, delimiter=",")
        if suffix == ".tsv":
            return _read_csv_or_tsv(path, delimiter="\t")
        raise ValueError(f"Unsupported annotation file: {path}")

    @staticmethod
    def _load_from_root(root: Path) -> List[Dict[str, Any]]:
        if not root.exists():
            raise FileNotFoundError(f"Data root not found: {root}")

        samples: List[Dict[str, Any]] = []
        for mp4_path in sorted(root.rglob("*.mp4")):
            txt_path = mp4_path.with_suffix(".txt")
            if not txt_path.exists():
                continue
            text = txt_path.read_text(encoding="utf-8", errors="ignore").strip()
            if not text:
                continue
            samples.append(
                {
                    "video_path": str(mp4_path),
                    "text": text,
                }
            )
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        row = self.samples[idx]
        video_path = row.get("video_path", row.get("video", ""))
        text = row.get("label", row.get("text", ""))
        instruction = row.get("instruction", self.default_instruction)

        if not video_path:
            raise ValueError(f"Missing video path in sample index {idx}")
        if not text:
            raise ValueError(f"Missing text label in sample index {idx}")

        return {
            "video_path": str(video_path),
            "text": str(text),
            "instruction": str(instruction),
        }

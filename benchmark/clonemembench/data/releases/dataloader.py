import json
from pathlib import Path
from dataclasses import dataclass


@dataclass
class CloneMemSample:
    """A single persona's memory benchmark data."""
    person_name: str
    person_id: str
    context: list[dict]      # List of digital traces (diary, chat, memo, etc.)
    questions: list[dict]    # List of QA items

    @property
    def num_traces(self) -> int:
        return len(self.context)

    @property
    def num_questions(self) -> int:
        return len(self.questions)


class CloneMemDataset:
    """CloneMem Benchmark Dataset."""

    def __init__(self, path: str, context_len: str = "100k"):
        """
        Args:
            path: Path to dataset directory
            context_len: Context length size, either "100k" or "500k"
        """
        self.path = Path(path)
        self.context_len = context_len
        self.samples: list[CloneMemSample] = []
        self._load()

    def _load(self):
        level_dir = self.path / self.context_len
        if not level_dir.exists():
            raise FileNotFoundError(f"Directory not found: {level_dir}")

        for json_file in level_dir.glob("*.json"):
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            sample = CloneMemSample(
                person_name=data["person_name"],
                person_id=data["person_id"],
                context=data["context"],
                questions=data["questions"]
            )
            self.samples.append(sample)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> CloneMemSample:
        return self.samples[idx]

    def __iter__(self):
        return iter(self.samples)

    def get_all_questions(self) -> list[dict]:
        """Get all questions across all personas."""
        questions = []
        for sample in self.samples:
            for q in sample.questions:
                q_with_meta = {
                    "person_name": sample.person_name,
                    "person_id": sample.person_id,
                    **q
                }
                questions.append(q_with_meta)
        return questions

    def stats(self) -> dict:
        """Get dataset statistics."""
        total_traces = sum(s.num_traces for s in self.samples)
        total_questions = sum(s.num_questions for s in self.samples)
        return {
            "context length": self.context_len,
            "num_personas": len(self.samples),
            "total_traces": total_traces,
            "total_questions": total_questions,
            "avg_traces_per_persona": total_traces / len(self.samples) if self.samples else 0,
            "avg_questions_per_persona": total_questions / len(self.samples) if self.samples else 0,
        }


def load_clonemem(path: str, context_len: str = "100k") -> CloneMemDataset:
    """
    Load CloneMem benchmark dataset.

    Args:
        path: Path to dataset directory
        context_len: "100k" or "500k"

    Returns:
        CloneMemDataset object

    Example:
        >>> dataset = load_clonemem("./dataset", level="100k")
        >>> print(len(dataset))
        >>> for sample in dataset:
        ...     print(sample.person_name, sample.num_questions)
    """
    return CloneMemDataset(path, context_len)

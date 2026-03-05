# CloneMem Dataset Format
---

## Directory Structure

```
releases/
├── README.md           # This file
├── dataloader.py       # Python dataloader utility
├── 100k/               # Short context (~100k tokens)
│   └── *.json
└── 500k/               # Long context (~500k tokens)
    └── *.json
```

---

## Quick Start

```python
from dataloader import load_clonemem

dataset = load_clonemem("./releases", context_len="100k")

for sample in dataset:
    print(f"{sample.person_name}: {sample.num_traces} traces, {sample.num_questions} questions")
```

---

## Data Schema

Each JSON file represents a single persona:

```json
{
  "person_name": "Hao Lin",
  "person_id": "5857744e-07fc-4db3-a86f-46b1b956641b",
  "context": [...],
  "questions": [...]
}
```

---

## Digital Traces (`context`)

Non-conversational digital traces spanning 1-3 years of a persona's life.

```json
{
  "id": "53ecdbb5-5219-4b8b-a213-9036766f013f",
  "medium": "search_history",
  "event_date": "2022-09-03T20:30:00",
  "content": "# Search History\n\n**2022-09-03**\n\n20:32 - Psychological reasons for self-doubt\n..."
}
```

The `medium` field indicates the type of digital trace, e.g., `diary`, `chat_private`, `chat_group`, `memo`, `email`, `social_media`, `search_history`, etc.

---

## Questions (`questions`)

Evaluation items with ground-truth answers and evidence.

```json
{
  "id": "47f73c71-f425-41d7-ac94-54d090eb4a04",
  "question": "Do you remember two years ago, when you were frantically searching for...",
  "question_type": "comparison",
  "question_time": "2024-09-28T22:00:00",
  "answer": "Haha, I can't believe you still remember that...",
  "dimension": "opinion",
  "digital_trace_ids": ["ca6b55ff-...", "0164aa56-..."],
  "evidence": [
    {
      "statement": "Lin Hao frantically searched for terms like 'product manager salary'...",
      "digital_trace_ids": ["ca6b55ff-45cd-4c0e-9ce8-e9c19e74a17c"]
    }
  ],
  "choices": [
    {"id": "A", "text": "Actually, it was the retrospective of that major promotion..."},
    {"id": "B", "text": "..."},
    {"id": "C", "text": "..."},
    {"id": "D", "text": "..."},
    {"id": "E", "text": "Cannot be determined based on available information"}
  ],
  "correct_choice_id": "D"
}
```

### Question Types

| Type | Description |
|------|-------------|
| `single_point_factual` | Retrieve explicit information at a specific time point |
| `comparison` | Compare between two time points |
| `trajectory` | Characterize evolution over extended periods |
| `pattern` | Identify recurring behaviors |
| `causal` | Trace event chains explaining changes |
| `counterfactual` | Consider alternative decision outcomes |
| `inferential` | Form judgments from scattered information |
| `unanswerable` | Recognize insufficient evidence |

### Dimensions

- `experience` — Factual events and activities
- `emotion` — Emotional states and psychological changes
- `opinion` — Beliefs, preferences, and evolving viewpoints
"""LAMBADA target splitting and greedy final-word accuracy arithmetic."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def test_lambada_target_split_basic():
    from src.common.dataset import lambada_target_split

    context, target = lambada_target_split("the cat sat on the mat")
    assert context == "the cat sat on the"
    assert target == " mat"


def test_lambada_target_split_trailing_whitespace_and_punctuation():
    from src.common.dataset import lambada_target_split

    context, target = lambada_target_split("she opened the door.  \n")
    assert context == "she opened the"
    assert target == " door."


def test_lambada_target_split_without_context():
    from src.common.dataset import lambada_target_split

    context, target = lambada_target_split("word")
    assert context == ""
    assert target == "word"


torch = pytest.importorskip("torch")

from torch import nn  # noqa: E402


class FixedLogitsModel(nn.Module):
    """Predicts token (input + 1) mod vocab at every position."""

    def __init__(self, vocab_size: int = 16):
        super().__init__()
        self.vocab_size = vocab_size
        self.config = type("Config", (), {"pad_token_id": 0})()

    def forward(self, input_ids=None, attention_mask=None):
        batch, width = input_ids.shape
        logits = torch.zeros(batch, width, self.vocab_size)
        successor = (input_ids + 1) % self.vocab_size
        logits.scatter_(2, successor.unsqueeze(-1), 10.0)
        return type("Output", (), {"logits": logits})()


def example(tokens: list[int], target_len: int) -> dict:
    return {
        "input_ids": torch.tensor(tokens, dtype=torch.long),
        "target_len": target_len,
    }


def test_accuracy_counts_exact_target_matches():
    from src.common.metrics import evaluate_lambada_accuracy

    model = FixedLogitsModel()
    device = torch.device("cpu")
    correct = example([1, 2, 3, 4], target_len=1)      # 3 -> 4 predicted
    also_correct = example([5, 6, 7, 8], target_len=2)  # 6->7, 7->8
    wrong = example([1, 2, 9], target_len=1)            # model predicts 3, not 9
    accuracy, count = evaluate_lambada_accuracy(
        model, [correct, also_correct, wrong], device, batch_size=2
    )
    assert count == 3
    assert accuracy == pytest.approx(2 / 3)


def test_accuracy_partial_target_match_is_incorrect():
    from src.common.metrics import evaluate_lambada_accuracy

    model = FixedLogitsModel()
    # First target token matches (3->4) but the second does not (4->9).
    partial = example([2, 3, 4, 9], target_len=2)
    accuracy, count = evaluate_lambada_accuracy(
        model, [partial], torch.device("cpu"), batch_size=4
    )
    assert count == 1
    assert accuracy == 0.0


def test_accuracy_is_padding_invariant():
    from src.common.metrics import evaluate_lambada_accuracy

    model = FixedLogitsModel()
    device = torch.device("cpu")
    short = example([1, 2, 3], target_len=1)
    long = example([4, 5, 6, 7, 8, 9], target_len=1)
    # Batched together, the short example is right-padded; the score must
    # match the unbatched evaluations.
    together, _ = evaluate_lambada_accuracy(model, [short, long], device, batch_size=2)
    alone_short, _ = evaluate_lambada_accuracy(model, [short], device, batch_size=1)
    alone_long, _ = evaluate_lambada_accuracy(model, [long], device, batch_size=1)
    assert together == pytest.approx((alone_short + alone_long) / 2)

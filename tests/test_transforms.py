"""Tests for transforms."""

from anymodel.utils._transforms import apply_transforms, middle_out


def test_middle_out_no_truncation():
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hi"},
    ]
    result = middle_out(messages, 128000)
    assert len(result) == 2


def test_middle_out_truncates():
    messages = [
        {"role": "system", "content": "System prompt"},
        {"role": "user", "content": "A" * 4000},
        {"role": "assistant", "content": "B" * 4000},
        {"role": "user", "content": "C" * 4000},
        {"role": "assistant", "content": "D" * 4000},
        {"role": "user", "content": "Latest message"},
    ]
    result = middle_out(messages, 500)
    assert len(result) < len(messages)
    # System prompt preserved
    assert result[0]["role"] == "system"


def test_apply_transforms_unknown():
    messages = [{"role": "user", "content": "Hi"}]
    result = apply_transforms(["unknown-transform"], messages)
    assert result == messages

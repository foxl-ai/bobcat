"""Host-side correctness feedback for a frozen, licensed decision curriculum.

This interface separates sampled feedback from the oracle label used by the
supervised control. It is a testable information boundary, not a security
sandbox: the host necessarily possesses the gold annotations.
"""

from __future__ import annotations

import copy

import torch


class DecisionBandit:
    def __init__(self, rows):
        if not rows or any(row["supervision"] != "hard_label" for row in rows):
            raise ValueError("Bandit correctness needs categorical outcomes, not score means.")
        self._rows = rows
        self._targets = tuple(row["target_index"] for row in rows)
        if any(type(y) is not int or not 0 <= y < len(row["option_token_ids"])
               for y, row in zip(self._targets, rows, strict=True)):
            raise ValueError("An observed category lies outside the request.")

    def __len__(self):
        return len(self._rows)

    def observation(self, index):
        row = self._rows[index]
        return copy.deepcopy({
            key: row[key] for key in ("input_ids", "option_token_ids", "candidate_ids")
        })

    def provenance(self, index):
        row = self._rows[index]
        return {key: row[key] for key in ("id", "group_id", "language", "task", "input_sha256")
                if key in row}

    def step(self, index, actions):
        count = len(self._rows[index]["option_token_ids"])
        if (actions.dtype != torch.long or actions.ndim != 1 or not actions.numel()
                or bool(((actions < 0) | (actions >= count)).any())):
            raise ValueError("The environment accepts only legal sampled actions.")
        return (actions == self._targets[index]).float().detach()

    def reveal_for_supervised_control(self, index):
        return self._targets[index]

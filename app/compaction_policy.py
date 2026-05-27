from __future__ import annotations

from dataclasses import dataclass

from app.session import SessionData


@dataclass(slots=True)
class CompactionPolicy:
    keep_rounds: int
    min_round_delta_for_resummarize: int
    level: int


def get_compaction_level(session: SessionData) -> int:
    try:
        return max(0, int(session.extra.get("compaction_level", 0)))
    except (TypeError, ValueError):
        return 0


def build_compaction_policy(session: SessionData) -> CompactionPolicy:
    level = get_compaction_level(session)

    if level <= 0:
        return CompactionPolicy(keep_rounds=6, min_round_delta_for_resummarize=2, level=0)
    if level == 1:
        return CompactionPolicy(keep_rounds=5, min_round_delta_for_resummarize=2, level=1)
    if level == 2:
        return CompactionPolicy(keep_rounds=4, min_round_delta_for_resummarize=3, level=2)
    return CompactionPolicy(keep_rounds=3, min_round_delta_for_resummarize=4, level=level)

from __future__ import annotations

from scripts.agent_performance.build_agent_performance_report import _select_rows_for_report


def test_select_rows_prefers_latest_run_id_when_present() -> None:
    rows = [
        {"run_id": "old", "experiment": "baseline", "episode_index": 1, "score": 10},
        {"run_id": "old", "experiment": "baseline", "episode_index": 2, "score": 11},
        {"run_id": "new", "experiment": "baseline", "episode_index": 1, "score": 21},
        {"run_id": "new", "experiment": "baseline", "episode_index": 2, "score": 22},
    ]
    selected, meta = _select_rows_for_report(rows, run_id="new", experiment="baseline")
    assert [int(r.get("score", 0)) for r in selected] == [21, 22]
    assert str(meta.get("method")) == "run_id"


def test_select_rows_fallback_uses_latest_episode_segment() -> None:
    rows = [
        {"episode_index": 1, "score": 5},
        {"episode_index": 2, "score": 6},
        {"episode_index": 1, "score": 15},
        {"episode_index": 2, "score": 16},
        {"episode_index": 3, "score": 17},
    ]
    selected, meta = _select_rows_for_report(rows, run_id="", experiment="")
    assert [int(r.get("score", 0)) for r in selected] == [15, 16, 17]
    assert str(meta.get("method")) == "latest_segment_fallback"


def test_select_rows_no_silent_fallback_when_run_id_mismatch_and_rows_have_ids() -> None:
    rows = [
        {"run_id": "old", "experiment": "baseline", "episode_index": 1, "score": 10},
        {"run_id": "old", "experiment": "baseline", "episode_index": 2, "score": 11},
    ]
    selected, meta = _select_rows_for_report(rows, run_id="new", experiment="baseline")
    assert selected == []
    assert str(meta.get("method")) == "run_id_no_match"


def test_select_rows_run_id_uses_latest_monotonic_segment_when_episode_index_resets() -> None:
    rows = [
        {"run_id": "r1", "experiment": "baseline", "episode_index": 1, "score": 10},
        {"run_id": "r1", "experiment": "baseline", "episode_index": 2, "score": 11},
        {"run_id": "r1", "experiment": "baseline", "episode_index": 1, "score": 20},
        {"run_id": "r1", "experiment": "baseline", "episode_index": 2, "score": 21},
        {"run_id": "r1", "experiment": "baseline", "episode_index": 3, "score": 22},
    ]
    selected, meta = _select_rows_for_report(rows, run_id="r1", experiment="baseline")
    assert [int(r.get("score", 0)) for r in selected] == [20, 21, 22]
    assert str(meta.get("method")) == "run_id"
    assert int(meta.get("selected_row_count", 0)) == 3
    assert int(meta.get("run_row_count_total", 0)) == 5

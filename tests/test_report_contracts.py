from __future__ import annotations

from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import numpy as np

from scripts.reporting.contracts import (
    CONTRACTS,
    REPORT_FAMILY_AGENT_PERFORMANCE,
    REPORT_FAMILY_PHASE3_COMPARE,
    REPORT_FAMILY_REPORTS_HUB,
    REPORT_FAMILY_TRAINING_INPUT,
    canonical_dir,
)


ROOT = Path(__file__).resolve().parents[1]


def _run(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *args],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=120,
    )


def _assert_required_latest(family: str) -> None:
    out_dir = canonical_dir(ROOT, family)
    contract = CONTRACTS[family]
    assert out_dir.exists()
    for rel in contract.required_latest_files:
        assert (out_dir / rel).exists(), f"missing {family} latest file: {rel}"


def _write_training_artifact_dir(path: Path) -> None:
    (path / "eval_logs").mkdir(parents=True, exist_ok=True)
    (path / "checkpoints").mkdir(parents=True, exist_ok=True)
    (path / "metadata.json").write_text(
        (
            '{"latest_run_id":"r_contract","requested_total_timesteps":500000,"actual_total_timesteps":507904,'
            '"config":{"env_count":8,"n_steps":1024,"batch_size":256,"n_epochs":8}}'
        ),
        encoding="utf-8",
    )
    (path / "training_trace.jsonl").write_text(
        '{"run_id":"r_contract","run_started_at_unix_s":1700000000,"requested_total_timesteps":500000,"actual_total_timesteps":507904}\n',
        encoding="utf-8",
    )
    (path / "eval_logs" / "evaluations_trace.jsonl").write_text(
        '{"run_id":"r_contract","step":200000,"mean_reward":4000.0,"mean_score":130.0,"eval_run_index":1}\n'
        '{"run_id":"r_contract","step":400000,"mean_reward":4050.0,"mean_score":132.0,"eval_run_index":2}\n',
        encoding="utf-8",
    )
    vec = SimpleNamespace(
        obs_rms=SimpleNamespace(
            mean=np.array([0.1, -0.2, 0.3], dtype=np.float64),
            var=np.array([0.11, 0.12, 0.13], dtype=np.float64),
            count=1000.0,
        ),
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=0.995,
        epsilon=1e-8,
        norm_obs=True,
        norm_reward=True,
    )
    (path / "vecnormalize.pkl").write_bytes(pickle.dumps(vec))
    (path / "checkpoints" / "step_200000_steps.zip").write_text("model", encoding="utf-8")
    (path / "checkpoints" / "step_vecnormalize_200000_steps.pkl").write_bytes(pickle.dumps(vec))


def test_training_input_contract_outputs_and_canonical_paths() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        artifact = Path(tmp) / "train_artifacts"
        _write_training_artifact_dir(artifact)

        out_dir = canonical_dir(ROOT, REPORT_FAMILY_TRAINING_INPUT)
        out_dir.mkdir(parents=True, exist_ok=True)
        # clean stale latest files to make assertions strict.
        for rel in CONTRACTS[REPORT_FAMILY_TRAINING_INPUT].required_latest_files:
            (out_dir / rel).unlink(missing_ok=True)

        result = _run(
            [
                "scripts/training_input/build_training_input_report.py",
                "--artifact-dir",
                str(artifact),
                "--out-dir",
                str(out_dir),
                "--tag",
                "latest",
                "--retain-stamped",
                "2",
            ]
        )
        assert result.returncode == 0, result.stderr or result.stdout
        result = _run(
            [
                "scripts/training_input/build_training_input_timeline.py",
                "--artifact-dir",
                str(artifact),
                "--out-dir",
                str(out_dir),
                "--tag",
                "latest",
                "--retain-stamped",
                "2",
            ]
        )
        assert result.returncode == 0, result.stderr or result.stdout
        result = _run(
            [
                "scripts/training_input/build_training_input_visuals.py",
                "--in-dir",
                str(out_dir),
                "--out-dir",
                str(out_dir),
                "--tag",
                "latest",
                "--retain-stamped",
                "2",
            ]
        )
        assert result.returncode == 0, result.stderr or result.stdout
        _assert_required_latest(REPORT_FAMILY_TRAINING_INPUT)


def test_agent_performance_contract_outputs_and_canonical_paths() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        artifact = Path(tmp) / "agent_artifacts"
        artifact.mkdir(parents=True, exist_ok=True)
        (artifact / "metadata.json").write_text('{"latest_run_id":"r_agent"}', encoding="utf-8")
        run_log = Path(tmp) / "run_session_log.jsonl"
        run_log.write_text(
            '{"episode_index":1,"score":10,"death_reason":"body","mode":"ppo","train_total_steps":1024,'
            '"interventions_pct":5.0,"interventions_delta":2,"decisions_delta":100,"risk_total":20,'
            '"stuck_episode_delta":0,"loop_escape_activations_total":0,"generated_at_unix_s":1700000000}\n',
            encoding="utf-8",
        )
        out_dir = canonical_dir(ROOT, REPORT_FAMILY_AGENT_PERFORMANCE)
        out_dir.mkdir(parents=True, exist_ok=True)
        for rel in CONTRACTS[REPORT_FAMILY_AGENT_PERFORMANCE].required_latest_files:
            (out_dir / rel).unlink(missing_ok=True)

        result = _run(
            [
                "scripts/agent_performance/build_agent_performance_report.py",
                "--artifact-dir",
                str(artifact),
                "--run-log",
                str(run_log),
                "--out-dir",
                str(out_dir),
                "--tag",
                "latest",
            ]
        )
        assert result.returncode == 0, result.stderr or result.stdout
        result = _run(
            [
                "scripts/agent_performance/build_agent_performance_visuals.py",
                "--in-dir",
                str(out_dir),
                "--out-dir",
                str(out_dir),
                "--tag",
                "latest",
            ]
        )
        assert result.returncode == 0, result.stderr or result.stdout
        _assert_required_latest(REPORT_FAMILY_AGENT_PERFORMANCE)


def test_phase3_compare_contract_outputs_and_canonical_paths() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        ppo_root = Path(tmp) / "ppo"
        left = ppo_root / "left_exp"
        right = ppo_root / "right_exp"
        for exp in (left, right):
            (exp / "eval_logs").mkdir(parents=True, exist_ok=True)
            (exp / "metadata.json").write_text(
                (
                    '{"latest_run_id":"r1","actual_total_timesteps":500000,'
                    '"config":{"env_count":8,"n_steps":1024,"batch_size":256,"n_epochs":8},'
                    '"training_episode_summary":{"episodes_total":5,"score_summary":{"mean":100,"p90":140,"best":160,"last":120},"deaths":{"body":2,"wall":1},'
                    '"episode_steps_summary":{"mean":100}},'
                    '"latest_eval_trace":{"mean_score":130.0},"best_eval_score":4200.0,"last_eval_score":4100.0}'
                ),
                encoding="utf-8",
            )
            (exp / "eval_logs" / "evaluations_trace.jsonl").write_text(
                '{"run_id":"r1","step":200000,"mean_reward":4100.0,"mean_score":120.0}\n',
                encoding="utf-8",
            )
            (exp / "last_model.zip").write_text("x", encoding="utf-8")
            (exp / "vecnormalize.pkl").write_text("x", encoding="utf-8")

        out_dir = canonical_dir(ROOT, REPORT_FAMILY_PHASE3_COMPARE)
        out_dir.mkdir(parents=True, exist_ok=True)
        for rel in CONTRACTS[REPORT_FAMILY_PHASE3_COMPARE].required_latest_files:
            (out_dir / rel).unlink(missing_ok=True)

        result = _run(
            [
                "scripts/phase3_compare/build_model_agent_compare_report.py",
                "--left-exp",
                "left_exp",
                "--right-exp",
                "right_exp",
                "--ppo-root",
                str(ppo_root),
                "--out-dir",
                str(out_dir),
                "--tag",
                "latest",
            ]
        )
        assert result.returncode == 0, result.stderr or result.stdout
        result = _run(
            [
                "scripts/phase3_compare/build_model_agent_compare_visuals.py",
                "--in-dir",
                str(out_dir),
                "--out-dir",
                str(out_dir),
                "--tag",
                "latest",
            ]
        )
        assert result.returncode == 0, result.stderr or result.stdout
        _assert_required_latest(REPORT_FAMILY_PHASE3_COMPARE)


def test_reports_hub_contract_outputs() -> None:
    out_dir = canonical_dir(ROOT, REPORT_FAMILY_REPORTS_HUB)
    out_dir.mkdir(parents=True, exist_ok=True)
    for rel in CONTRACTS[REPORT_FAMILY_REPORTS_HUB].required_latest_files:
        (out_dir / rel).unlink(missing_ok=True)
    result = _run(
        [
            "scripts/reporting/build_reports_hub.py",
            "--artifacts-root",
            "artifacts",
            "--out-dir",
            str(out_dir),
        ]
    )
    assert result.returncode == 0, result.stderr or result.stdout
    _assert_required_latest(REPORT_FAMILY_REPORTS_HUB)


def test_cli_invalid_out_dir_fails_non_zero_and_deterministic_message() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        bad_out = Path(tmp) / "not_canonical"
        bad_out.mkdir(parents=True, exist_ok=True)
        result = _run(
            [
                "scripts/training_input/build_training_input_report.py",
                "--out-dir",
                str(bad_out),
            ]
        )
        assert result.returncode != 0
        joined = f"{result.stdout}\n{result.stderr}"
        assert "--out-dir must be canonical for training_input" in joined

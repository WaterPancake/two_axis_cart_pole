"""Frozen common-seed comparison for the single-axis controllers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from mc_pilco.single_axis_mc import load_single_axis_mc
from mc_pilco.single_axis_ppo import load_single_axis_ppo
from mc_pilco.single_axis_task import EPISODE_STEPS, TASK_ID, evaluate_policy

PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MC = PROJECT_DIR / "artifacts" / "single-axis" / "mc-pilco" / "checkpoint-final.pt"
DEFAULT_PPO = PROJECT_DIR / "artifacts" / "single-axis" / "ppo" / "checkpoint-final.zip"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_DIR.resolve()))
    except ValueError:
        return str(path)


def _training_metadata(path: Path) -> dict[str, object]:
    report_path = path.parent / "report.json"
    if not report_path.is_file():
        return {}
    report = json.loads(report_path.read_text(encoding="utf-8"))
    return {
        key: report.get(key)
        for key in (
            "method",
            "real_transitions",
            "imagined_transitions",
            "wall_seconds",
            "uses_expert_or_classical_controller",
        )
        if key in report
    }


def generate_single_axis_report(
    *,
    mc_checkpoint: Path = DEFAULT_MC,
    ppo_checkpoint: Path = DEFAULT_PPO,
    episodes_per_stratum: int = 50,
    output: Path | None = None,
) -> dict[str, object]:
    if episodes_per_stratum < 1:
        raise ValueError("episodes_per_stratum must be positive")
    mc = load_single_axis_mc(mc_checkpoint)
    ppo = load_single_axis_ppo(ppo_checkpoint)
    seed = 41_003
    report = {
        "task": TASK_ID,
        "protocol": {
            "seed": seed,
            "episodes_per_stratum": episodes_per_stratum,
            "episode_steps": EPISODE_STEPS,
            "shared_initial_state_generator": True,
            "runtime_expert_or_fallback": False,
        },
        "mc_pilco": {
            "checkpoint": _display_path(mc_checkpoint),
            "checkpoint_sha256": _sha256(mc_checkpoint),
            **_training_metadata(mc_checkpoint),
            "evaluation": evaluate_policy(
                mc.control,
                seed=seed,
                episodes_per_stratum=episodes_per_stratum,
            ),
        },
        "ppo": {
            "checkpoint": _display_path(ppo_checkpoint),
            "checkpoint_sha256": _sha256(ppo_checkpoint),
            **_training_metadata(ppo_checkpoint),
            "evaluation": evaluate_policy(
                ppo.control,
                seed=seed,
                episodes_per_stratum=episodes_per_stratum,
            ),
        },
    }
    destination = output or PROJECT_DIR / "artifacts" / "single-axis" / "comparison.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report

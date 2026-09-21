from pathlib import Path
import tempfile

from mc_pilco.experiment import (
    ExperimentConfig,
    PROJECT_DIR,
    checkpoint_frame_skip,
    load_dynamics,
    load_policy,
    train,
)


def test_tiny_training_run_saves_deployable_artifacts() -> None:
    artifacts = PROJECT_DIR / "artifacts"
    with tempfile.TemporaryDirectory(dir=artifacts) as temporary:
        output = Path(temporary) / "run"
        metrics = train(
            ExperimentConfig(
                seed=2,
                initial_rollouts=2,
                trial_rollouts=1,
                rollout_steps=4,
                iterations=1,
                max_gp_points=8,
                gp_steps=1,
                policy_updates=1,
                particles=4,
                imagination_horizon=2,
                output_dir=output,
            )
        )
        assert metrics["dataset_size"] == 12
        assert (output / "transitions.npz").is_file()
        assert (output / "metrics.json").is_file()
        checkpoint = output / "checkpoint.pt"
        assert checkpoint.is_file()
        policy = load_policy(checkpoint)
        assert policy.hidden_sizes == (32, 32)
        model = load_dynamics(checkpoint)
        assert model.train_x.shape == (8, 12)
        assert model.output_dim == 5
        assert checkpoint_frame_skip(checkpoint) == 5

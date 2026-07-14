"""Environment/back-end interfaces for the two-axis cart-pole."""

from envs.mujoco_interface import ArrowKeyController, TwoAxisInvertedPendulum

__all__ = ["ArrowKeyController", "TwoAxisInvertedPendulum"]


def __getattr__(name: str):
    """Lazily expose optional Gymnasium integration without requiring it."""

    if name in {"TwoAxisCartPoleEnv", "register_env"}:
        from envs.gymnasium_wrapper import TwoAxisCartPoleEnv, register_env

        globals()["TwoAxisCartPoleEnv"] = TwoAxisCartPoleEnv
        globals()["register_env"] = register_env
        return globals()[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

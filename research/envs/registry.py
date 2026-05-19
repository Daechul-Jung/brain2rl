"""
Environment registry — make_env(env_id, **kwargs) factory.

Supported env_id strings:

  Gymnasium / MuJoCo (any standard Gymnasium ID):
    "HalfCheetah-v4", "Ant-v4", "Hopper-v4", ...

  ManiSkill3 manipulation tasks (pick-and-place family):
    "PickCube-v1"       — pick a cube off a table
    "PushCube-v1"       — push a cube to a goal
    "StackCube-v1"      — stack one cube on another
    "PullCube-v1"       — pull a cube to a goal
    "PlaceSphere-v1"    — place a sphere in a bowl
    Any other registered ManiSkill env ID

  Custom OpenArm MuJoCo:
    "OpenArm-v0"

To add a new environment:
  1. Add an elif branch below (or extend _MANISKILL_IDS).
  2. Return a gymnasium.Env-compatible object.
  3. Add a test in tests/test_04_envs.py.

ManiSkill 3.0 notes:
  - obs is returned as torch.Tensor (N, obs_dim) directly — no wrapper needed for state mode.
  - Rewards and dones are also tensors; our agents handle both numpy and torch inputs.
  - Pass num_envs= to run parallel envs (default 1).
"""

from __future__ import annotations

import gymnasium as gym


# ManiSkill3 pick-and-place tasks we support
MANISKILL_TASKS = {
    "PickCube-v1",
    "PushCube-v1",
    "StackCube-v1",
    "PullCube-v1",
    "PlaceSphere-v1",
    "PickCubeSO100-v1",
    "TwoRobotPickCube-v1",
    "TwoRobotStackCube-v1",
}


def make_env(env_id: str, **kwargs) -> gym.Env:
    """
    Create and return a Gymnasium-compatible environment.

    Args:
        env_id  : environment identifier string (see module docstring)
        **kwargs: forwarded to the environment constructor

    Raises:
        ImportError  if the required library is not installed
        ValueError   if env_id is unrecognised
    """
    if env_id == "OpenArm-v0":
        return _make_openarm(**kwargs)

    if env_id in MANISKILL_TASKS or _looks_like_maniskill(env_id):
        return _make_maniskill(env_id, **kwargs)

    # Default: standard Gymnasium (covers all MuJoCo locomotion envs)
    try:
        return gym.make(env_id, **kwargs)
    except gym.error.Error as e:
        raise ValueError(
            f"Unknown env_id '{env_id}'. "
            "Valid options: any Gymnasium ID, "
            f"{sorted(MANISKILL_TASKS)}, 'OpenArm-v0'."
        ) from e


def _looks_like_maniskill(env_id: str) -> bool:
    """Heuristic: IDs ending in -v1/-v2 with known ManiSkill prefixes."""
    prefixes = ("Pick", "Push", "Stack", "Pull", "Place", "Lift", "Peg", "Open", "Close",
                "Assemble", "Pour", "Plug", "Faucet", "Drawer", "Cabinet")
    return any(env_id.startswith(p) for p in prefixes)


def _make_maniskill(env_id: str, **kwargs) -> gym.Env:
    try:
        import mani_skill.envs  # noqa: F401 — triggers env registration
    except ImportError as e:
        raise ImportError(
            "ManiSkill3 is not installed. Install with: pip install mani_skill"
        ) from e

    num_envs = kwargs.pop("num_envs", 1)
    obs_mode = kwargs.pop("obs_mode", "state")
    render_mode = kwargs.pop("render_mode", None)

    env = gym.make(
        env_id,
        obs_mode=obs_mode,
        num_envs=num_envs,
        render_mode=render_mode,
        **kwargs,
    )
    return env


def _make_openarm(**kwargs) -> gym.Env:
    import sys, os as _os
    sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), "..", ".."))
    try:
        from models.rl.envs.openarm_mj_env import OpenArmMjEnv
    except ImportError as e:
        raise ImportError(
            "OpenArm MuJoCo env failed to import. "
            "Check that external/openarm_mujoco/ XML files exist."
        ) from e
    _root = _os.path.join(_os.path.dirname(__file__), "..", "..")
    default_xml = _os.path.abspath(_os.path.join(
        _root, "external", "openarm_mujoco", "v1", "openarm_bimanual_pick_place_scene.xml"
    ))
    kwargs.setdefault("xml_path", default_xml)
    return OpenArmMjEnv(**kwargs)


def maniskill_available() -> bool:
    """Return True if ManiSkill3 is installed and the basic API works."""
    try:
        import mani_skill.envs  # noqa: F401
        import gymnasium as gym
        # Quick smoke test
        env = gym.make("PickCube-v1", obs_mode="state", num_envs=1)
        env.close()
        return True
    except Exception:
        return False

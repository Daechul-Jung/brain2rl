from typing import Tuple, Dict, Any


class TrackingEnvWrapper:
    """Wraps a mocap tracking task to expose required interface."""
    def __init__(self, env, ref_horizon: int = 5):
        # TODO: store env and settings
        pass

    @property
    def proprio_dim(self) -> int:
        # TODO
        pass

    @property
    def ref_dim(self) -> int:
        # TODO
        pass

    @property
    def act_dim(self) -> int:
        # TODO
        pass

    def reset(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """TODO: return obs dict {"proprio": ..., "ref": ...}, info."""
        # TODO
        pass

    def step(self, action) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """TODO: return (obs, reward, terminated, truncated, info)."""
        # TODO
        pass


class ComplementaryTaskEnvWrapper:
    """Wraps a transfer/joint task using the same robot/physics."""
    def __init__(self, env):
        # TODO
        pass

    @property
    def obs_dim(self) -> int:
        # TODO
        pass

    @property
    def act_dim(self) -> int:
        # TODO
        pass

    def reset(self):
        # TODO
        pass

    def step(self, action):
        # TODO
        pass
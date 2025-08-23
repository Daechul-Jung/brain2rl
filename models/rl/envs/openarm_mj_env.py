import numpy as np, mujoco
from gymnasium import spaces

class OpenArmMjEnv:
    """Minimal MuJoCo env wrapper for OpenArm """
    def __init__(self, xml_path: str, horizon=300, render=False, action_scale=0.03):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data  = mujoco.MjData(self.model)
        self.horizon = horizon
        self.render  = render
        self.action_scale = action_scale
        self.t = 0

        # Assume all actuators are joints you control (adjust if needed)
        self.nu = self.model.nu
        self.nq = self.model.nq
        self.nv = self.model.nv

        # Use actuator ctrlrange if provided
        if self.model.actuator_ctrlrange.size == 2*self.nu:
            lo = self.model.actuator_ctrlrange[:,0].astype(np.float32)
            hi = self.model.actuator_ctrlrange[:,1].astype(np.float32)
            lo = np.where(np.isfinite(lo), lo, -1.0)
            hi = np.where(np.isfinite(hi), hi,  1.0)
            lo = np.maximum(lo, -1.0); hi = np.minimum(hi, 1.0)
        else:
            lo = -np.ones(self.nu, np.float32)
            hi =  np.ones(self.nu, np.float32)
        self.action_space = spaces.Box(low=lo, high=hi, dtype=np.float32)

        obs_hi = np.inf*np.ones(self.nq + self.nv, np.float32)
        self.observation_space = spaces.Box(-obs_hi, obs_hi, dtype=np.float32)

        # Optional viewer
        self.viewer = None
        if render:
            try:
                from mujoco import viewer
                self.viewer = viewer.launch_passive(self.model, self.data)
            except Exception:
                self.viewer = None

        # cache address of your task object (for randomization)
        self.cup_addr = None
        try:
            j = self.model.joint('cup_freejoint')
            self.cup_addr = slice(j.qposadr, j.qposadr+7)  # (x,y,z,w,x,y,z)
        except Exception:
            pass

    def _get_obs(self):
        return np.concatenate([self.data.qpos.copy(), self.data.qvel.copy()]).astype(np.float32)

    def reset(self, seed=None, options=None):
        if seed is not None: np.random.seed(seed)
        self.t = 0
        # randomize cup XY (if present)
        if self.cup_addr is not None:
            x = np.random.uniform(0.25, 0.45)
            y = np.random.uniform(-0.10, 0.10)
            self.data.qpos[self.cup_addr] = [x, y, 0.75, 1, 0, 0, 0]
        # reset arm
        self.data.qpos[:self.nq] = 0.0
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high) * self.action_scale
        self.data.ctrl[:self.nu] = action
        mujoco.mj_step(self.model, self.data)
        self.t += 1

        obs = self._get_obs()
        # Example reward: keep joints near zero (replace with your task’s reward)
        reward = -float(np.linalg.norm(self.data.qpos[:self.nu]))
        terminated = False
        truncated  = self.t >= self.horizon
        info = {}
        if self.render and self.viewer:
            self.viewer.sync()
        return obs, reward, terminated, truncated, info

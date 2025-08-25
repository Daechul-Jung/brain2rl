import numpy as np, mujoco
from gymnasium import spaces
import imageio.v2 as imageio


class OpenArmMjEnv:
    """OpenArm (left arm) reaching a cup; optional camera observations."""
    def __init__(self, xml_path: str, horizon=300, render=False,
                 action_scale=0.03, camera=None, camera_size=(256,256), camera_in_info = True):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data  = mujoco.MjData(self.model)
        self.horizon = horizon
        self.render  = render
        self.action_scale = action_scale
        self.t = 0

        ##############################
        cams = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_CAMERA, i) for i in range(self.model.ncam)]
        print("Cameras:", cams)
        ##########################


        # --- pick left-arm actuators ---
        self.left_actuators = []
        for i in range(self.model.nu):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if name and name.startswith("left_"):
                self.left_actuators.append(i)
        self.left_actuators = np.asarray(self.left_actuators, dtype=np.int32)
        assert len(self.left_actuators) > 0, "No 'left_' actuators found."

        # sizes
        self.nu = len(self.left_actuators)
        self.nq = self.model.nq
        self.nv = self.model.nv

        # action space (use actuator ctrlrange if provided)
        if self.model.actuator_ctrlrange.size == 2*self.model.nu:
            lo_all = self.model.actuator_ctrlrange[:,0].astype(np.float32)
            hi_all = self.model.actuator_ctrlrange[:,1].astype(np.float32)
            lo_all = np.where(np.isfinite(lo_all), lo_all, -1.0)
            hi_all = np.where(np.isfinite(hi_all), hi_all,  1.0)


        else:
            lo_all = -np.ones(self.model.nu, np.float32)
            hi_all =  np.ones(self.model.nu, np.float32)

        lo = lo_all[self.left_actuators]
        hi = hi_all[self.left_actuators]
        self.action_space = spaces.Box(low=lo, high=hi, dtype=np.float32)

        # observation: qpos, qvel, ee(xyz), cup(xyz)
        obs_dim = self.nq + self.nv + 3 + 3
        obs_hi = np.inf*np.ones(obs_dim, np.float32)
        self.observation_space = spaces.Box(-obs_hi, obs_hi, dtype=np.float32)

        # cache ids for sites/joints
        self.sid_left_ee = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "left_ee")
        self.sid_cup_top = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "cup_top")
        self.jid_cup_free = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cup_freejoint")
        assert self.sid_left_ee >= 0 and self.sid_cup_top >= 0 and self.jid_cup_free >= 0, \
            "Missing left_ee/cup_top/cup_freejoint (check your XML edits)."

        # rendering
        self.viewer = None
        if render:
            try:
                from mujoco import viewer
                self.viewer = viewer.launch_passive(self.model, self.data)
            except Exception:
                self.viewer = None

        # offscreen camera for CV
        self.camera_name = camera  
        self.camera_size = tuple(camera_size)
        self.renderer = None
        self.camera_in_info = camera_in_info
        if self.camera_name is not None:
            if isinstance(self.camera_name, str):
                self.cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, self.camera_name)
                assert self.cam_id >= 0, f"Camera '{self.camera_name}' not found in XML."
            else:
                self.cam_id = int(self.camera_name)
            self.renderer = mujoco.Renderer(self.model, *self.camera_size)
    # ------------- helpers -------------
    def _cup_qpos_slice(self):
        jadr = self.model.jnt_qposadr[self.jid_cup_free]
        return slice(jadr, jadr+7)  

    def _ee(self):  return self.data.site_xpos[self.sid_left_ee].copy()
    def _cup(self): return self.data.site_xpos[self.sid_cup_top].copy()

    def _get_pixels(self):
        """Return HxWx3 uint8 RGB from the configured camera, or None."""
        if self.renderer is None:
            return None
        self.renderer.update_scene(self.data, camera=self.cam_id)
        rgb = self.renderer.render()                  
        return rgb
    
    def _get_obs(self):
        base = [self.data.qpos.ravel(), self.data.qvel.ravel(), self._ee(), self._cup()]
        obs = np.concatenate(base).astype(np.float32)
        if self.renderer is not None:
            self.renderer.update_scene(self.data, camera=self.camera_name)
            rgb = self.renderer.render()  ### Camera size (256, 256, 3)
        

        return obs
    
    def calculate_reward(self, info):
        """
        Should include pose estimation, distance between cup and arm, 
        """

        reward = 0

        return reward
    
    # ------------- RL API -------------
    def reset(self, seed=None, options=None):
        if seed is not None: np.random.seed(seed)
        self.t = 0
        mujoco.mj_resetData(self.model, self.data)

        # randomize cup XY (freejoint qpos = [x y z qw qx qy qz])
        qsl = self._cup_qpos_slice()
        x = np.random.uniform(0.30, 0.48)
        y = np.random.uniform(-0.12, 0.12)
        self.data.qpos[qsl] = [x, y, 0.06, 1, 0, 0, 0]

        # zero velocities; settle
        self.data.qvel[:] = 0.0
        for _ in range(10):
            mujoco.mj_step(self.model, self.data)
        obs = self._get_obs()
        info = {}
        if self.camera_in_info:
            pix = self._get_pixels()
            if pix is not None:
                info["pixels"] = pix
                imageio.imwrite("camera_view.png", pix)
        return obs, info

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high) * self.action_scale
        self.data.ctrl[self.left_actuators] = action
        mujoco.mj_step(self.model, self.data)
        self.t += 1

        ee  = self._ee()
        cup = self._cup()
        dist = np.linalg.norm(ee - cup)


        # reward = self.calculate_reward()
        # simple shaping towards the cup
        reward = -dist
        reward += 0.1*(0.2 - np.clip(dist, 0, 0.2))

        terminated = dist < 0.03
        truncated  = self.t >= self.horizon
        info = {"dist": dist}

        if self.render and self.viewer:
            self.viewer.sync()

        return self._get_obs(), float(reward), terminated, truncated, info


    
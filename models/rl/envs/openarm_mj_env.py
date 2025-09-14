import numpy as np, mujoco
from gymnasium import spaces
import imageio.v2 as imageio
from models.rl.envs.vision_utils import *
import os


class OpenArmMjEnv:
    """OpenArm (left arm) reaching a cup; optional camera observations."""
    def __init__(self, xml_path: str, horizon=300, render=False,
                 action_scale=0.03, camera=None, camera_size=(256,256), camera_in_info = False, 
                 vision_rewards_weight = 0.4, physics_rewards_weight = 0.6, target_cup = 'cup1'):
        
        ### camera="left_wrist_cam"
        self.model = mujoco.MjModel.from_xml_path(xml_path)  ### robot model
        self.data = mujoco.MjData(self.model) ### MuJoCo model data 
        self.horizon = horizon  ### Horizon for each episode 
        self.print_component()
        self.render = render
        self.action_scale = action_scale
        self.t = 0  ### inital time step 
        self.camera_size = camera_size
        ###### Weights for rewards components #####
        self.vision_weight = vision_rewards_weight
        self.physic_weight = physics_rewards_weight

        self.target = target_cup  ### Target cup to reach
        self.cup_geom_ids = self._geoms_matching(["cup"])
        #### Setting reward calculator #####
        self.reward_calc = RewardCalculator(
            vision_weight=vision_rewards_weight,
            physics_weight=physics_rewards_weight,
            cup_geom_ids=self.cup_geom_ids,
            camera_size=self.camera_size,
        )
        ### Show cup geometries ###
        print("Cup geoms:", [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, g) for g in self.cup_geom_ids])

        #### Cup detector 
        self.vision_detector = CupDetector(camera_size=self.camera_size,
                                        cup_geom_ids=self.cup_geom_ids)

        ##############################
        cams = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_CAMERA, i) for i in range(self.model.ncam)]
        print("Cameras:", cams)
        ##########################


        # --- pick left-arm actuators ---
        self.left_actuators = []
        for i in range(self.model.nu): # for number of actuator
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) # convert actuator id to name
            if name and name.startswith("left_"):
                self.left_actuators.append(i) # extract only left actuators
        self.left_actuators = np.asarray(self.left_actuators, dtype=np.int32)
        assert len(self.left_actuators) > 0, "No 'left_' actuators found."

        # sizes
        self.nu = len(self.left_actuators) # actuators
        self.nq = self.model.nq # Generalized positions of DOF
        self.nv = self.model.nv # Generalized velocity DOF

        # action space (use actuator ctrlrange if provided)
        if self.model.actuator_ctrlrange.size == 2*self.model.nu:
            lo_all = self.model.actuator_ctrlrange[:,0].astype(np.float32) ### All zeros 
            hi_all = self.model.actuator_ctrlrange[:,1].astype(np.float32) ### Some of them are 0.044
            lo_all = np.where(np.isfinite(lo_all), lo_all, -1.0)
            hi_all = np.where(np.isfinite(hi_all), hi_all,  1.0)
            tiny = (hi_all - lo_all) < 1e-8
            lo_all[tiny] = -1.0
            hi_all[tiny] =  1.0
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
        
        self._ee_kind, self._ee_id = None, -1
        candidates = [
            ("site", "left_ee"),
            ("site", "openarm_left_ee"),
            ("body", "openarm_left_hand_tcp"),
            ("body", "openarm_left_hand"),
        ]
        for kind, nm in candidates:
            objtype = mujoco.mjtObj.mjOBJ_SITE if kind == "site" else mujoco.mjtObj.mjOBJ_BODY
            eid = mujoco.mj_name2id(self.model, objtype, nm)
            if eid >= 0:
                self._ee_kind, self._ee_id = kind, eid
                break

        self.sid_cup_top  = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE,  "cup_top")
        self.jid_cup_free = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cup_freejoint")

        # optional: keep backward-compatible alias if other code uses sid_left_ee
        self.sid_left_ee = self._ee_id if self._ee_kind == "site" else -1

        assert self._ee_id >= 0 and self.sid_cup_top >= 0 and self.jid_cup_free >= 0, \
            "Missing EE (site left_ee/openarm_left_ee or body openarm_left_hand[_tcp]), cup_top, or cup_freejoint."

        # rendering
        self.viewer = None
        if render:
            try:
                from mujoco import viewer
                self.viewer = viewer.launch_passive(self.model, self.data)
            except Exception:
                self.viewer = None


        self.camera_name = camera
        self.camera_size = tuple(camera_size)
        self.camera_in_info = camera_in_info
        self.renderer = None
        self.cam_id = None
        # if self.camera_name is not None:
        #     if isinstance(self.camera_name, str):
        #         self.cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, self.camera_name)
        #         assert self.cam_id >= 0, (
        #             f"Camera '{self.camera_name}' not found. Available: "
        #             f"{[mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_CAMERA, i) for i in range(self.model.ncam)]}"
        #         )
        #     else:
        #         self.cam_id = int(self.camera_name)
        #     self.renderer = mujoco.Renderer(self.model, *self.camera_size)
            
    # ------------- helpers -------------
    def print_component(self):
        self.geom_id_cup_wall = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "cup_wall")
        self.geom_id_cup_bottom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "cup_bottom")
        self.sid_cup_top    = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "cup_top")

        # finger slide joints (for fallback aperture calc)
        self.jid_f1 = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "openarm_left_finger_joint1")
        self.jid_f2 = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "openarm_left_finger_joint2")
        self.qadr_f1 = self.model.jnt_qposadr[self.jid_f1] if self.jid_f1 >= 0 else -1
        self.qadr_f2 = self.model.jnt_qposadr[self.jid_f2] if self.jid_f2 >= 0 else -1
        print(f'data site_pos: \n {self.data.site_xpos}')

        print(f'geom_id_cup_wall: {self.geom_id_cup_wall}') # is it index of data?

    def cup_dimensions(self):
        # Prefer the wall geom (cylinder)
        r, h = None, None
        if self.gid_cup_wall >= 0:
            size = self.model.geom_size[self.gid_cup_wall]
            # CYLINDER: [radius, half_length]
            r = float(size[0])
            h = float(2.0 * size[1])
        elif self.gid_cup_bottom >= 0: 
            size = self.model.geom_size[self.gid_cup_bottom]
            r = float(size[0])
            h = float(2.0 * size[1])
        return r, h 


    def _cup_qpos_slice(self):
        jadr = self.model.jnt_qposadr[self.jid_cup_free]
        return slice(jadr, jadr+7)  

    def _ee(self):
        # site_xpos for sites; xpos for bodies
        return (self.data.site_xpos[self._ee_id] if self._ee_kind == "site"
                else self.data.xpos[self._ee_id]).copy()
    def _cup(self): return self.data.site_xpos[self.sid_cup_top].copy() ### Cup top position

    def _get_pixels(self, with_seg=False):
        if self.renderer is None:
            return None if not with_seg else (None, None)
        self.renderer.update_scene(self.data, camera=self.cam_id)  # use mujoco model data and camera id 
        if with_seg:
            try:
                rgb, seg = self.renderer.render(segmentation=True)
            except TypeError:
                # older mujoco: returns depth not seg; fallback to rgb only
                rgb, seg = self.renderer.render(), None
            return rgb, seg
        else:
            return self.renderer.render()

    def _get_obs(self):

        base = [self.data.qpos.ravel(), self.data.qvel.ravel(), self._ee(), self._cup()]
        obs = np.concatenate(base).astype(np.float32)
        if self.renderer is not None and self.cam_id is not None:
            self.renderer.update_scene(self.data, camera=self.cam_id)
            _ = self.renderer.render()  # warm cache

        return obs
    
    def _geoms_matching(self, substr_list):
        out = []
        for gid in range(self.model.ngeom): # geom id 
            nm = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, gid)
            if nm and all(s in nm for s in substr_list):
                out.append(gid)
        return np.asarray(out, dtype=np.int32)
    
    def calculate_reward(self, ee, cup):
        """
        Should include pose estimation, distance between cup and arm, 
        how to calculate pose estimation and check whether fingers are open enough to grab the cup. 
        """
        dist = float(np.linalg.norm(ee - cup))
        dist_reward = -dist
        # dist_reward += 0.1* (0.2 - np.clip(dist, 0, 0.2)) ## Small amount of bonus rewards when it is closer than 0.2
        finger_reward = self.evaluate_finger(ee)
        pose_reward = self.evaluate_pose(ee)
        reward = dist_reward #+ finger_reward + pose_reward

        return reward

    def evaluate_finger(self, ee):
        return 
    
    def evaluate_pose(self, ee):
        finger_pose = self.evaluate_finger(ee)
        return 
    
    # ------------- RL API -------------
    
    def reset(self, seed=5678, options=None):

        if seed is not None: np.random.seed(seed)
        self.t = 0
        mujoco.mj_resetData(self.model, self.data)

        qsl = self._cup_qpos_slice()
        x = np.random.uniform(0.30, 0.48)
        y = np.random.uniform(0.08, 0.20)
        self.data.qpos[qsl] = [x, y, 0.06, 1, 0, 0, 0]

        # zero velocities; settle
        self.data.qvel[:] = 0.0
        for _ in range(10):
            mujoco.mj_step(self.model, self.data)
        obs = self._get_obs()
        info = {}

        return obs, info

    def step(self, action):

        # Scaling the action 
        action = np.clip(action, self.action_space.low, self.action_space.high) * self.action_scale

        ## put action in model data control part for left_actuators part
        self.data.ctrl[self.left_actuators] = action

        # Take MuJoCo step
        mujoco.mj_step(self.model, self.data)
        self.t += 1
  
        ee, cup = self._ee(), self._cup() ### need to get and calculate the pose of end effector and cup
        dist = float(np.linalg.norm(ee - cup))
        

        total_reward = self.calculate_reward(ee, cup)
        terminated = dist < 0.03
        truncated  = self.t >= self.horizon

        info = {
            "dist": dist,
            "total_reward": float(total_reward),
        }


        if self.render and self.viewer:
            self.viewer.sync()

        return self._get_obs(), float(total_reward), terminated, truncated, info

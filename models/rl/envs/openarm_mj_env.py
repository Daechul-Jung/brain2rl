import numpy as np, mujoco
from gymnasium import spaces
import imageio.v2 as imageio
from models.rl.envs.vision_utils import *
import os


class OpenArmMjEnv:
    """OpenArm (left arm) reaching a cup; optional camera observations."""
    def __init__(self, xml_path: str, horizon=300, render=False,
                 action_scale=0.03, camera='left_wrist_cam', camera_size=(256,256), camera_in_info = True, 
                 vision_rewards_weight = 0.4, physics_rewards_weight = 0.6, target_cup = 'cup1'):
        
        self.model = mujoco.MjModel.from_xml_path(xml_path)  ### robot model
        self.data = mujoco.MjData(self.model) ### MuJoCo model data 
        self.horizon = horizon  ### Horizon for each episode 
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
        print("Cup geoms:", [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, g)
                            for g in self.cup_geom_ids])

        #### Cup detector 
        self.vision_detector = CupDetector(camera_size=self.camera_size,
                                        cup_geom_ids=self.cup_geom_ids)

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
        self.sid_cup_top = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "cup1_top")
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

    def _ee(self):  return self.data.site_xpos[self.sid_left_ee].copy() ### End effector position
    def _cup(self): return self.data.site_xpos[self.sid_cup_top].copy() ### Cup top position

    def _get_pixels(self, with_seg=False):
        if self.renderer is None:
            return None if not with_seg else (None, None)
        self.renderer.update_scene(self.data, camera=self.cam_id)  # use id
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
        if self.renderer is not None:
            self.renderer.update_scene(self.data, camera=self.cam_id) 
            _ = self.renderer.render()
        return obs
    
    def _geoms_matching(self, substr_list):
        out = []
        for gid in range(self.model.ngeom):
            nm = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, gid)
            if nm and all(s in nm for s in substr_list):
                out.append(gid)
        return np.asarray(out, dtype=np.int32)
    
    def calculate_reward(self, info):
        """
        Should include pose estimation, distance between cup and arm, 
        """

        reward = 0

        return reward
    
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
        if self.camera_in_info:
            pix = self._get_pixels()
            if pix is not None:
                info["pixels"] = pix
                imageio.imwrite("camera_view.png", pix)
        return obs, info

    def step(self, action):
        # 1) physics
        action = np.clip(action, self.action_space.low, self.action_space.high) * self.action_scale
        self.data.ctrl[self.left_actuators] = action
        mujoco.mj_step(self.model, self.data)
        self.t += 1

        ee, cup = self._ee(), self._cup()
        dist = float(np.linalg.norm(ee - cup))

        # 2) render once (RGB + segmentation)
        rgb, seg = (None, None)
        if self.renderer is not None and self.camera_in_info:
            rgb, seg = self._get_pixels(with_seg=True)
################################## Added for verifying
            if self.renderer is not None and (self.t % 50 == 0):
                cid = self.cam_id
                cam_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_CAMERA, cid)
                cam_body = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, self.model.cam_bodyid[cid])
                pos = self.data.cam_xpos[cid].copy()
                R   = self.data.cam_xmat[cid].reshape(3,3).copy()
                print("optical_axis:", -R[:,2]) 
                print(f"[cam] using={cam_name} on body={cam_body} pos={pos} optical_axis={-R[:,2]}")
###############################
        # 3) see if the cup is even in view (fast probe)
        if seg is not None and self.cup_geom_ids.size and (self.t % 50 == 0):
            ids = np.unique(seg if seg.ndim == 2 else seg[..., 0])
            seen = np.intersect1d(ids, self.cup_geom_ids)
            print(f"[t={self.t}] cup geom ids visible: {seen.tolist()}")

        # 4) reward (now vision uses seg)
        total_reward, vis_info = self.reward_calc.calculate_total_reward(
            image=rgb if rgb is not None else np.zeros((1,1,3), dtype=np.uint8),
            ee_pos=ee, cup_pos=cup, target_cup=self.target, seg=seg
        )

        # 5) detection + optional overlay (debug)
        detected = {}
        if rgb is not None:
            detected = self.vision_detector.detect_cups(rgb, seg=seg)
            if (self.t % 100) == 0:
                os.makedirs("debug", exist_ok=True)
                overlay = self.vision_detector.visualize_detection(rgb, detected) if detected else rgb
                imageio.imwrite(os.path.join("debug", f"detect_{self.t:06d}.png"), overlay)

        # 6) termination
        terminated = dist < 0.03
        truncated  = self.t >= self.horizon

        # 7) info
        info = {
            "dist": dist,
            "physics_reward": float(vis_info.get("physics_reward", -dist)),
            "vision_reward": float(vis_info.get("vision_reward", 0.0)),
            "total_reward": float(total_reward),
            "vision_info": vis_info.get("vision_info", {}),
            "detected_cups": int(len(detected)),
        }
        if self.camera_in_info and rgb is not None:
            info["pixels"] = rgb

        if self.render and self.viewer:
            self.viewer.sync()

        return self._get_obs(), float(total_reward), terminated, truncated, info

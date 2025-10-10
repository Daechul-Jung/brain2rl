import numpy as np, mujoco
from gymnasium import spaces
import imageio.v2 as imageio
from models.rl.envs.vision_utils import *
from models.rl.envs.rewards_utils import *
import os


class OpenArmMjEnv:
    """OpenArm (left arm) reaching a cup; optional camera observations."""
    def __init__(self, 
                 xml_path: str, 
                 horizon=300, 
                 render=False,
                 action_scale=0.03, 
                 camera=None, 
                 camera_size=(256,256), 
                 camera_in_info = False, 

                 target_cup = 'cup1',
                 goal_mode: str = 'left_to_right', 
                 goal_offset: float = 0.20,  ## meters along +x (or -x)
                 success_radius:float = 0.03, ## cup within this of goal
                 reach_weight: float=1.0,
                 grasp_bonus: float = 1.0,
                 transport_weight: float = 8.0, ## progress shaping weight
                 hold_bonus_weight: float = 0.5, ## bonus while grasping& moving toward
                 success_bonus: float = 10.0,
                 action_l2_weight: float = 0.001,
                 time_penalty: float = 0.001):
        
        ### camera="left_wrist_cam"
        self.model = mujoco.MjModel.from_xml_path(xml_path)  ### robot model
        self.data = mujoco.MjData(self.model) ### MuJoCo model data 
        self.horizon = horizon  ### Horizon for each episode 
        self.print_component()
        self.render = render
        self.action_scale = action_scale
        self.t = 0  ### inital time step 

        self.goal_mode = goal_mode
        self.goal_offset = float(goal_offset)
        self.success_radius = float(success_radius)
        self.reach_w = float(reach_weight)
        self.grasp_bonus = float(grasp_bonus)
        self.transport_w = float(transport_weight)
        self.hold_bonus_w = float(hold_bonus_weight)
        self.success_bonus = float(success_bonus)
        self.action_l2_w = float(action_l2_weight)
        self.time_penalty = float(time_penalty)

        self.camera_size = camera_size
        ###### Weights for rewards components #####


        self.target = target_cup  ### Target cup to reach
        self._discover_id()
        self.use_curriculum = True
        self.curr_stage = 0  # 0: reach, 1: grasp, 2: transport, 3: success
        self.stage_success_count = 0
        self.stage_success_needed = [20, 20, 30]  # how many “good” steps before advancing
        self.stage_cfg = [
            {"reach":1.0, "grasp":0.0, "transport":0.0, "success":0.0},   # stage 0
            {"reach":0.5, "grasp":1.0, "transport":0.0, "success":0.0},   # stage 1
            {"reach":0.2, "grasp":1.0, "transport":1.0, "success":0.0},   # stage 2
            {"reach":0.1, "grasp":1.0, "transport":1.0, "success":1.0},   # stage 3
        ]

        self.left_actuators = []
        for i in range(self.model.nu):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if name and name.startswith("left_"):
                self.left_actuators.append(i)
        self.left_actuators = np.asarray(self.left_actuators, dtype=np.int32)
        assert len(self.left_actuators) > 0, "No 'left_' actuators found."


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

        # action space
        if self.model.actuator_ctrlrange.size == 2*self.model.nu:
            lo_all = self.model.actuator_ctrlrange[:,0].astype(np.float32)
            hi_all = self.model.actuator_ctrlrange[:,1].astype(np.float32)
            lo_all = np.where(np.isfinite(lo_all), lo_all, -1.0)
            hi_all = np.where(np.isfinite(hi_all), hi_all,  1.0)
            tiny = (hi_all - lo_all) < 1e-8
            lo_all[tiny] = -1.0; hi_all[tiny] = 1.0
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

        # rendering
        self.viewer = None
        if render:
            try:
                from mujoco import viewer
                self.viewer = viewer.launch_passive(self.model, self.data)
            except Exception:
                self.viewer = None
        self.cup_start = None
        self.goal_pos = None 
        self.prev_cup_goal_dist = None
        self.prev_cup_pos = None 
            
    # ------------- helpers -------------
    def _discover_id(self,):
        # EE: prefer site 'left_ee' then fallbacks
        self._ee_kind, self._ee_id = None, -1
        for kind, nm in [("site","left_ee"), ("site","openarm_left_ee"),
                         ("body","openarm_left_hand_tcp"), ("body","openarm_left_hand")]:
            objtype = mujoco.mjtObj.mjOBJ_SITE if kind == "site" else mujoco.mjtObj.mjOBJ_BODY
            eid = mujoco.mj_name2id(self.model, objtype, nm)
            if eid >= 0:
                self._ee_kind, self._ee_id = kind, eid
                break
        assert self._ee_id >= 0, "Missing EE site/body."

        # cup site (top) & freejoint
        self.sid_cup_top  = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE,  "cup_top")
        self.jid_cup_free = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cup_freejoint")
        assert self.sid_cup_top >= 0 and self.jid_cup_free >= 0, "Missing cup_top site or cup_freejoint."

        # cup geoms for contact filtering
        self.cup_geom_ids = []
        for gid in range(self.model.ngeom):
            nm = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, gid)
            if nm and "cup" in nm: self.cup_geom_ids.append(gid)
        self.cup_geom_ids = np.asarray(self.cup_geom_ids, dtype=np.int32)
        assert len(self.cup_geom_ids) > 0, "No cup geoms found."

        # finger joints (for aperture)
        self.jid_f1 = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "openarm_left_finger_joint1")
        self.jid_f2 = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "openarm_left_finger_joint2")
        self.qadr_f1 = self.model.jnt_qposadr[self.jid_f1] if self.jid_f1 >= 0 else -1
        self.qadr_f2 = self.model.jnt_qposadr[self.jid_f2] if self.jid_f2 >= 0 else -1

        # cache EE site id if it's a site
        self.sid_left_ee = self._ee_id if self._ee_kind == "site" else -1

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

    def _ee(self) -> np.ndarray:
        return (self.data.site_xpos[self._ee_id] if self._ee_kind == "site"
                else self.data.xpos[self._ee_id]).copy()

    def _cup(self) -> np.ndarray:
        return self.data.site_xpos[self.sid_cup_top].copy()

    def _gripper_aperture(self) -> float:
        """Fallback: aperture = |q_f1 - q_f2| if both slide joints exist."""
        if self.qadr_f1 >= 0 and self.qadr_f2 >= 0:
            return float(abs(self.data.qpos[self.qadr_f1] - self.data.qpos[self.qadr_f2]))
        return 0.0

    def _contacts_with_cup(self) -> List[Tuple[int,int]]:
        """Return list of (geom1, geom2) contacts where either is a cup geom."""
        out = []
        ncon = self.data.ncon
        for ci in range(ncon):
            con = self.data.contact[ci]
            if con.geom1 in self.cup_geom_ids or con.geom2 in self.cup_geom_ids:
                out.append((con.geom1, con.geom2))
        return out

    def _is_grasping(self) -> bool:
        """Heuristic: at least 2 contacts with cup AND aperture within a 'closing' band."""
        contacts = self._contacts_with_cup()
        if len(contacts) < 2:
            return False
        ap = self._gripper_aperture()
        # tune these two numbers to your gripper kinematics
        return 0.002 < ap < 0.020

    # ------------------- RL API -------------------

    def _get_obs(self):
        base = [self.data.qpos.ravel(), self.data.qvel.ravel(), self._ee(), self._cup()]
        return np.concatenate(base).astype(np.float32)

    def reset(self, seed=5678, options=None):
        if seed is not None: np.random.seed(seed)
        self.t = 0
        mujoco.mj_resetData(self.model, self.data)
        
        # randomize cup pose on table
        qsl = self._cup_qpos_slice()
        x = np.random.uniform(0.30, 0.48)
        y = np.random.uniform(0.08, 0.20)
        self.data.qpos[qsl] = [x, y, 0.06, 1, 0, 0, 0]

        # settle
        self.data.qvel[:] = 0.0
        for _ in range(10):
            mujoco.mj_step(self.model, self.data)

        # record start & define goal on x-axis
        self.cup_start = self._cup()
        direction = +1.0 if self.goal_mode == "left_to_right" else -1.0
        self.goal_pos = self.cup_start.copy()
        self.goal_pos[0] += direction * self.goal_offset
        direction = +1.0 if self.goal_mode == "left_to_right" else -1.0
        axis = "y"  # <<< try "y" instead of "x"
        self.goal_pos = self.cup_start.copy()
        if axis == "x":
            self.goal_pos[0] += direction * self.goal_offset
        else:
            self.goal_pos[1] += direction * self.goal_offset

        # shaping memory
        self.prev_cup_goal_dist = np.linalg.norm(self.cup_start - self.goal_pos)
        self.prev_cup_pos = self.cup_start.copy()

        obs = self._get_obs()
        info = {"goal_pos": self.goal_pos.copy()}
        return obs, info

    def step(self, action):
        # scale & apply to left actuators only
        action = np.clip(action, self.action_space.low, self.action_space.high) * self.action_scale
        self.data.ctrl[self.left_actuators] = action

        mujoco.mj_step(self.model, self.data)
        self.t += 1

        ee = self._ee()
        cup = self._cup()

        # reward
        r, shaped = self._calculate_reward(ee, cup, action)

        # success if cup at goal (no need to force release)
        cup_to_goal = np.linalg.norm(cup - self.goal_pos)
        terminated = cup_to_goal < self.success_radius
        truncated  = self.t >= self.horizon

        info = {
            "dist_ee_cup": float(np.linalg.norm(ee - cup)),
            "cup_to_goal": float(cup_to_goal),
            "grasping": bool(shaped["is_grasping"]),
            "progress": float(shaped["progress"]),
            "total_reward": float(r),
            "goal_pos": self.goal_pos.copy(),
        }

        if self.render and self.viewer:
            self.viewer.sync()
        # print(f"EE {self._ee()}  CUP {self._cup()}  GOAL {self.goal_pos}  mode={self.goal_mode}")

        return self._get_obs(), float(r), terminated, truncated, info
    
    def _calculate_reward(self, ee, cup, action):
        """
        Components:
          - Reach:   -||ee - cup||
          - Grasp:   +bonus if grasping (contacts + aperture)
          - Progress: transport shaping: (prev_dist_to_goal - current_dist_to_goal)
          - Hold bonus: bonus * progress if grasping & cup is moving toward goal
          - Success bonus: once within success_radius
          - Small penalties: action L2 and time
        """
        shaped = {}

        # base components (same as before)
        dist_ee_cup = np.linalg.norm(ee - cup)
        r_reach_raw = -dist_ee_cup * self.reach_w

        grasping = self._is_grasping()
        r_grasp_raw = self.grasp_bonus if grasping else 0.0

        cup_to_goal = np.linalg.norm(cup - self.goal_pos)
        progress = self.prev_cup_goal_dist - cup_to_goal
        r_transport_raw = self.transport_w * progress
        r_hold_raw = self.hold_bonus_w * progress if grasping and progress > 0 else 0.0
        r_success_raw = self.success_bonus if cup_to_goal < self.success_radius else 0.0

        r_act = -self.action_l2_w * float(np.sum(np.square(action)))
        r_time = -self.time_penalty

        # curriculum masks
        if self.use_curriculum:
            cfg = self.stage_cfg[self.curr_stage]
        else:
            cfg = {"reach":1.0, "grasp":1.0, "transport":1.0, "success":1.0}

        # apply masks
        r_reach    = cfg["reach"]     * r_reach_raw
        r_grasp    = cfg["grasp"]     * r_grasp_raw
        r_transport= cfg["transport"] * r_transport_raw
        r_hold     = cfg["transport"] * r_hold_raw
        r_success  = cfg["success"]   * r_success_raw

        r_total = r_reach + r_grasp + r_transport + r_hold + r_success + r_act + r_time

        # stage advancement heuristic:
        good_signal = (
            (self.curr_stage == 0 and dist_ee_cup < 0.06) or
            (self.curr_stage == 1 and grasping) or
            (self.curr_stage == 2 and progress > 0.001) or
            (self.curr_stage >= 3 and cup_to_goal < self.success_radius)
        )
        if self.use_curriculum:
            if good_signal: 
                self.stage_success_count += 1
            else:
                self.stage_success_count = max(0, self.stage_success_count - 1)  # hysteresis
            # advance when sustained “good” behavior
            if self.curr_stage < len(self.stage_cfg) - 1 and \
            self.stage_success_count >= self.stage_success_needed[self.curr_stage]:
                self.curr_stage += 1
                self.stage_success_count = 0

        # book-keeping
        self.prev_cup_goal_dist = cup_to_goal
        self.prev_cup_pos = cup.copy()

        shaped.update({
            "is_grasping": grasping,
            "progress": progress,
            "curr_stage": self.curr_stage,
            "stage_success_count": self.stage_success_count
        })
        return r_total, shaped
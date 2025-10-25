"""
In for multiple tasks, I would import 'PickCube-v1', 'PullCube-v1', 'PushCube-v1' and 'StackCube-v1'
Sequence is Push, pull, pick and stack. I want to test two different types of sequences. 
One is sequences set by manually. Achieve one task and evaluate and move on to the next
"""
import sapien
from typing import Dict, Any, Optional
import torch
import torch.random
from transforms3d.euler import euler2quat
from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.agents.robots import SO100, Panda
from mani_skill.sensors.camera import CameraConfig, Camera
from mani_skill.utils.building import actors
from mani_skill.utils.sapien_utils import look_at
from mani_skill.utils.registration import register_env
from mani_skill.utils import common
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import Array
# from mani_skill.envs.tasks.tabletop import pick_cube, pull_cube, push_cube, stack_cube  ### These are just module 
from mani_skill.envs.tasks.tabletop import PickCubeEnv, PullCubeEnv, PushCubeEnv, StackCubeEnv  ### import for calculating rewards
from mani_skill.envs.sapien_env import BaseEnv


class combinedTask:
    def __init__(self):
        ...
        
@register_env(uid="Combined-v1", max_episode_steps=100, override=False)
class CombinedTaskEnv(BaseEnv):
    """
    In this environment, Agent would perform sequence of tasks and later thinking what to do first and plan for this. 
    Sequences can be represented as graph or other formats, for now I am thinking about two types: manual setting and let the agent think about the process
    """
    SUPPORTED_ROBOTS = ['so100', 'panda']
    goal_radius = 0.1
    cube_half_size = 0.02

    def __init__(self, *args, robot_uid = 'so100', robot_init_qpos_noise = 0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uid,**kwargs)

        self.pullEnv = PullCubeEnv(*args, robot_uids=robot_uid, robot_init_qpos_noise=self.robot_init_qpos_noise)
        self.pushEnv = PushCubeEnv(*args, robot_uids=robot_uid, robot_init_qpos_noise=self.robot_init_qpos_noise)
        self.pickEnv = PickCubeEnv(*args, robot_uids=robot_uid, robot_init_qpos_noise=self.robot_init_qpos_noise)
        self.stackEnv = StackCubeEnv(*args, robot_uids=robot_uid, robot_init_qpos_noise=self.robot_init_qpos_noise)
        self.stage = None ### Variable for determining 

    def _load_agent(self, options: Dict):
        super()._load_agent(options, sapien.Pose(p = [-0.615, 0, 0]))

    def _load_scene(self, options: Dict):
        self.cube_half_size = common.to_tensor([0.02] * 3, device= self.device)
        self.table_scene = TableSceneBuilder(
            env= self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        ## create cube
        self.CubeA = actors.build_cube(
            self.scene, 
            half_size=0.02,
            color= [1, 0, 0, 1],
            name='CubeA',
            initial_pose=sapien.Pose(p = [0, 0, 0.1])
        )

        ## create another cube for stacking
        self.CubeB = actors.build_cube(
            self.scene,
            half_size=0.02,
            color= [0, 1, 0, 1],
            name='CubeB',
            initial_pose=sapien.Pose(p = [1, 0, 0.1])
        )

        self.goal_region = actors.build_red_white_target(
            self.scene,
            radius =  self.goal_radius,
            thickness=1e-5,
            name = 'goal_region',
            add_collision=False,
            body_type='kinematic'
        )

    def evaluate(self):
        """
        Evaluate current situation and need to consider processes of tasks: algorithmize this part.
        """
        self.pickEnv.evaluate()
        self.pullEnv.evaluate()
        self.pushEnv.evaluate()
        self.stackEnv.evaluate()

    def _get_obs_extra(self):
        obs = dict(
            tcp_pose = self.agent.tcp.pose.raw_pose,
            goal_pos = self.goal_region.pose.p,
        )
        if self.obs_mode_struct.use_state:
            obs.update(
                obj_pose = self.obj.pose.raw_pose
            )
        return obs
    
    def compute_dense_reward(self, obs, action, info):
        """
        Checking the stage and calculating the rewards based on the stage. I would use just dense rewards rather than normalized one
        """
        
        return

    
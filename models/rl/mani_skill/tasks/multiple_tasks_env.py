"""
In for multiple tasks, I would import 'PickCube-v1', 'PullCube-v1', 'PushCube-v1' and 'StackCube-v1'
Sequence is Push, pull, pick and stack. I want to test two different types of sequences. 
One is sequences set by manually. Achieve one task and evaluate and move on to the next
"""
import sapien
from typing import Dict, Any, Optional
import mani_skill.envs.utils.randomization as randomization
import torch
import os, sys
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
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        
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
        self.stage = { ## initially using dictionary but using other data type for later
            "push": False,
            "pull": False,
            "pick": False,
            "stack": False,
        }

    def _load_agent(self, options: Dict):
        """
        Load the agent. However, think about how to replace this with VLA or my own model 
        """
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
        Each evaluate function returns info whether success, is_obj_placed, is_static, is_grasped
        """
        pick_info = self.pickEnv.evaluate()
        pull_info = self.pullEnv.evaluate()
        push_info = self.pushEnv.evaluate()
        stack_info = self.stackEnv.evaluate()

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            xyz = torch.zeros((b, 3))
            xyz[:, :2] = (
                torch.rand((b, 2)) * self.cube_spawn_half_size * 2
                - self.cube_spawn_half_size
            )
            xyz[:, 0] += self.cube_spawn_center[0]
            xyz[:, 1] += self.cube_spawn_center[1]

            xyz[:, 2] = self.cube_half_size
            qs = randomization.random_quaternions(b, lock_x=True, lock_y=True)
            self.cube.set_pose(Pose.create_from_pq(xyz, qs))

            goal_xyz = torch.zeros((b, 3))
            goal_xyz[:, :2] = (
                torch.rand((b, 2)) * self.cube_spawn_half_size * 2
                - self.cube_spawn_half_size
            )
            goal_xyz[:, 0] += self.cube_spawn_center[0]
            goal_xyz[:, 1] += self.cube_spawn_center[1]
            goal_xyz[:, 2] = torch.rand((b)) * self.max_goal_height + xyz[:, 2]
            self.goal_site.set_pose(Pose.create_from_pq(goal_xyz))

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
    
    def _compute_dense_reward(self, obs, action, info):
        """
        Checking the stage and calculating the rewards based on the stage. I would use just dense rewards rather than normalized one
        If the stage supposed to do is not completed but the agent tries to do other tasks, give them penalty
        """
        rewards = 0
        if info['stage'] == 'pull':
            rewards += self.pullEnv.compute_dense_reward(obs, action, info)
        
        elif info['stage'] == 'push':
            rewards = self.pushEnv.compute_dense_reward(obs, action, info)

        elif info['stage'] == 'pick':
            rewards = self.pickEnv.compute_dense_reward(obs, action, info)

        elif info['stage'] == 'stack':
            rewards = self.stackEnv.compute_dense_reward(obs, action, info)
            
        return rewards

    
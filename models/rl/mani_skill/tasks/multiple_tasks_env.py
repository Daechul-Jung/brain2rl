"""
In for multiple tasks, I would import 'PickCube-v1', 'PullCube-v1', 'PushCube-v1' and 'StackCube-v1'
Sequence is Push, pull, pick and stack. I want to test two different types of sequences. 
One is sequences set by manually. Achieve one task and evaluate and move on to the next
"""
import sapien
from typing import Dict, Any, Optional
import mani_skill.envs.utils.randomization as randomization
import torch
import numpy as np
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
    def __init__(self, *args, robot_uids = 'so100', robot_init_qpos_noise = 0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.SUPPORTED_ROBOTS = ['so100', 'panda']
        self.goal_radius = 0.1
        self.cube_half_size = 0.2
        self.cube_spawn_center = (0,0)
        self.cube_spawn_half_size = 0.05
        self.max_goal_height = 0.3
        if 'robot_uids' in kwargs:
            if kwargs['robot_uids'] != robot_uids:
                robot_uids = kwargs['robot_uids']
            kwargs.pop('robot_uids', None)
        super().__init__(*args, robot_uids=robot_uids,**kwargs)


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
        if self.stage['pick']:
            info = self._evaluate_pick()

        elif self.stage['pull']:
            info = self._evaluate_pull()

        elif self.stage['push']:
            info = self._evaluate_push()

        else:
            info = self._evaluate_stack()

        return 

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        if self.stage['pick']:
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
                print(xyz, self.cube_half_size)
                xyz[:, 2] = self.cube_half_size
                qs = randomization.random_quaternions(b, lock_x=True, lock_y=True)
                self.CubeA.set_pose(Pose.create_from_pq(xyz, qs))

                goal_xyz = torch.zeros((b, 3))
                goal_xyz[:, :2] = (
                    torch.rand((b, 2)) * self.cube_spawn_half_size * 2
                    - self.cube_spawn_half_size
                )
                goal_xyz[:, 0] += self.cube_spawn_center[0]
                goal_xyz[:, 1] += self.cube_spawn_center[1]
                goal_xyz[:, 2] = torch.rand((b)) * self.max_goal_height + xyz[:, 2]
                self.goal_site.set_pose(Pose.create_from_pq(goal_xyz))

        elif self.stage['pull']:
            with torch.device(self.device):
                b = len(env_idx)
                self.table_scene.initialize(env_idx)
                xyz = torch.zeros((b, 3))
                xyz[..., :2] = torch.rand((b, 2)) * 0.2 - 0.1
                xyz[..., 2] = self.cube_half_size
                q = [1, 0, 0, 0]

                obj_pose = Pose.create_from_pq(p=xyz, q=q)
                self.obj.set_pose(obj_pose)

                target_region_xyz = xyz - torch.tensor([0.1 + self.goal_radius, 0, 0])

                target_region_xyz[..., 2] = 1e-3
                self.goal_region.set_pose(
                    Pose.create_from_pq(
                        p=target_region_xyz,
                        q=euler2quat(0, np.pi / 2, 0),
                    )
                )
        elif self.stage['stack']:
            with torch.device(self.device):
                b = len(env_idx)
                self.table_scene.initialize(env_idx)

                xyz = torch.zeros((b, 3))
                xyz[:, 2] = 0.02
                xy = torch.rand((b, 2)) * 0.2 - 0.1
                region = [[-0.1, -0.2], [0.1, 0.2]]
                sampler = randomization.UniformPlacementSampler(
                    bounds=region, batch_size=b, device=self.device
                )
                radius = torch.linalg.norm(torch.tensor([0.02, 0.02])) + 0.001
                cubeA_xy = xy + sampler.sample(radius, 100)
                cubeB_xy = xy + sampler.sample(radius, 100, verbose=False)

                xyz[:, :2] = cubeA_xy
                qs = randomization.random_quaternions(
                    b,
                    lock_x=True,
                    lock_y=True,
                    lock_z=False,
                )
                self.CubeA.set_pose(Pose.create_from_pq(p=xyz.clone(), q=qs))

                xyz[:, :2] = cubeB_xy
                qs = randomization.random_quaternions(
                    b,
                    lock_x=True,
                    lock_y=True,
                    lock_z=False,
                )
                self.CubeB.set_pose(Pose.create_from_pq(p=xyz, q=qs))

        elif self.stage['push']:
            with torch.device(self.device):
                b = len(env_idx)
                # when using scene builders, you must always call .initialize on them so they can set the correct poses of objects in the prebuilt scene
                # note that the table scene is built such that z=0 is the surface of the table.
                self.table_scene.initialize(env_idx)

                # here we write some randomization code that randomizes the x, y position of the cube we are pushing in the range [-0.1, -0.1] to [0.1, 0.1]
                xyz = torch.zeros((b, 3))
                xyz[..., :2] = torch.rand((b, 2)) * 0.2 - 0.1
                xyz[..., 2] = self.cube_half_size
                q = [1, 0, 0, 0]
               
                obj_pose = Pose.create_from_pq(p=xyz, q=q)
                self.obj.set_pose(obj_pose)

                # here we set the location of that red/white target (the goal region). In particular here, we set the position to be in front of the cube
                # and we further rotate 90 degrees on the y-axis to make the target object face up
                target_region_xyz = xyz + torch.tensor([0.1 + self.goal_radius, 0, 0])
                # set a little bit above 0 so the target is sitting on the table
                target_region_xyz[..., 2] = 1e-3
                self.goal_region.set_pose(
                    Pose.create_from_pq(
                        p=target_region_xyz,
                        q=euler2quat(0, np.pi / 2, 0),
                    )
                )        
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
        Do I need to set reward to consider complete tasks in the right sequences? maybe yse, but how
        """
        rewards = 0
        if info['stage'] == 'pull':
            rewards = self._compute_dense_reward_pull(obs, action, info)
        
        elif info['stage'] == 'push':
            rewards = self._compute_dense_reward_push(obs, action, info)

        elif info['stage'] == 'pick':
            rewards = self._compute_dense_reward_pick(obs, action, info)

        elif info['stage'] == 'stack':
            rewards = self._compute_dense_reward_stack(obs, action, info)
            
        return rewards

    ######## From maniskill document ###################
    def _evaluate_pick(self):
        is_obj_placed = (
            torch.linalg.norm(self.goal_site.pose.p - self.cube.pose.p, axis=1)
            <= self.goal_thresh
        )
        is_grasped = self.agent.is_grasping(self.cube)
        is_robot_static = self.agent.is_static(0.2)
        return {
            "success": is_obj_placed & is_robot_static,
            "is_obj_placed": is_obj_placed,
            "is_robot_static": is_robot_static,
            "is_grasped": is_grasped,
        }

    def _evaluate_pull(self):
        is_obj_placed = (
            torch.linalg.norm(
                self.obj.pose.p[..., :2] - self.goal_region.pose.p[..., :2], axis=1
            )
            < self.goal_radius
        )

        return {
            "success": is_obj_placed,
        }

    def _evaluate_push(self):
        is_obj_placed = (
            torch.linalg.norm(
                self.obj.pose.p[..., :2] - self.goal_region.pose.p[..., :2], axis=1
            )
            < self.goal_radius
        ) & (self.obj.pose.p[..., 2] < self.cube_half_size + 5e-3)

        return {
            "success": is_obj_placed,
        }

    def _evaluate_stack(self):
        pos_A = self.cubeA.pose.p
        pos_B = self.cubeB.pose.p
        offset = pos_A - pos_B
        xy_flag = (
            torch.linalg.norm(offset[..., :2], axis=1)
            <= torch.linalg.norm(self.cube_half_size[:2]) + 0.005
        )
        z_flag = torch.abs(offset[..., 2] - self.cube_half_size[..., 2] * 2) <= 0.005
        is_cubeA_on_cubeB = torch.logical_and(xy_flag, z_flag)
        # NOTE (stao): GPU sim can be fast but unstable. Angular velocity is rather high despite it not really rotating
        is_cubeA_static = self.cubeA.is_static(lin_thresh=1e-2, ang_thresh=0.5)
        is_cubeA_grasped = self.agent.is_grasping(self.cubeA)
        success = is_cubeA_on_cubeB * is_cubeA_static * (~is_cubeA_grasped)
        return {
            "is_cubeA_grasped": is_cubeA_grasped,
            "is_cubeA_on_cubeB": is_cubeA_on_cubeB,
            "is_cubeA_static": is_cubeA_static,
            "success": success.bool(),
        }

    def _compute_dense_reward_pick(self, obs, action, info):
        tcp_to_obj_dist = torch.linalg.norm(
            self.cube.pose.p - self.agent.tcp_pose.p, axis=1
        )
        reaching_reward = 1 - torch.tanh(5 * tcp_to_obj_dist)
        reward = reaching_reward

        is_grasped = info["is_grasped"]
        reward += is_grasped

        obj_to_goal_dist = torch.linalg.norm(
            self.goal_site.pose.p - self.cube.pose.p, axis=1
        )
        place_reward = 1 - torch.tanh(5 * obj_to_goal_dist)
        reward += place_reward * is_grasped

        qvel = self.agent.robot.get_qvel()
        if self.robot_uids in ["panda", "widowxai"]:
            qvel = qvel[..., :-2]
        elif self.robot_uids == "so100":
            qvel = qvel[..., :-1]
        static_reward = 1 - torch.tanh(5 * torch.linalg.norm(qvel, axis=1))
        reward += static_reward * info["is_obj_placed"]

        reward[info["success"]] = 5
        return reward

    def _compute_dense_reward_push(self, obs, action, info):
        tcp_push_pose = Pose.create_from_pq(
            p=self.obj.pose.p
            + torch.tensor([-self.cube_half_size - 0.005, 0, 0], device=self.device)
        )
        tcp_to_push_pose = tcp_push_pose.p - self.agent.tcp.pose.p
        tcp_to_push_pose_dist = torch.linalg.norm(tcp_to_push_pose, axis=1)
        reaching_reward = 1 - torch.tanh(5 * tcp_to_push_pose_dist)
        reward = reaching_reward

        # compute a placement reward to encourage robot to move the cube to the center of the goal region
        # we further multiply the place_reward by a mask reached so we only add the place reward if the robot has reached the desired push pose
        # This reward design helps train RL agents faster by staging the reward out.
        reached = tcp_to_push_pose_dist < 0.01
        obj_to_goal_dist = torch.linalg.norm(
            self.obj.pose.p[..., :2] - self.goal_region.pose.p[..., :2], axis=1
        )
        place_reward = 1 - torch.tanh(5 * obj_to_goal_dist)
        reward += place_reward * reached

        # Compute a z reward to encourage the robot to keep the cube on the table
        desired_obj_z = self.cube_half_size
        current_obj_z = self.obj.pose.p[..., 2]
        z_deviation = torch.abs(current_obj_z - desired_obj_z)
        z_reward = 1 - torch.tanh(5 * z_deviation)
        # We multiply the z reward by the place_reward and reached mask so that
        #   we only add the z reward if the robot has reached the desired push pose
        #   and the z reward becomes more important as the robot gets closer to the goal.
        reward += place_reward * z_reward * reached

        # assign rewards to parallel environments that achieved success to the maximum of 3.
        reward[info["success"]] = 4
        return reward

    def _compute_dense_reward_pull(self, obs, action, info):
        tcp_pull_pos = self.obj.pose.p + torch.tensor(
            [self.cube_half_size + 2 * 0.005, 0, 0], device=self.device
        )
        tcp_to_pull_pose = tcp_pull_pos - self.agent.tcp.pose.p
        tcp_to_pull_pose_dist = torch.linalg.norm(tcp_to_pull_pose, axis=1)
        reaching_reward = 1 - torch.tanh(5 * tcp_to_pull_pose_dist)
        reward = reaching_reward

        reached = tcp_to_pull_pose_dist < 0.01
        obj_to_goal_dist = torch.linalg.norm(
            self.obj.pose.p[..., :2] - self.goal_region.pose.p[..., :2], axis=1
        )
        place_reward = 1 - torch.tanh(5 * obj_to_goal_dist)
        reward += place_reward * reached

        reward[info["success"]] = 3
        return reward

    def _compute_dense_reward_stack(self, obs, action, info):
        tcp_pose = self.agent.tcp.pose.p
        cubeA_pos = self.CubeA.pose.p
        cubeA_to_tcp_dist = torch.linalg.norm(tcp_pose - cubeA_pos, axis=1)
        reward = 2 * (1 - torch.tanh(5 * cubeA_to_tcp_dist))

        # grasp and place reward
        cubeA_pos = self.CubeA.pose.p
        cubeB_pos = self.CubeB.pose.p
        goal_xyz = torch.hstack(
            [cubeB_pos[:, 0:2], (cubeB_pos[:, 2] + self.cube_half_size[2] * 2)[:, None]]
        )
        cubeA_to_goal_dist = torch.linalg.norm(goal_xyz - cubeA_pos, axis=1)
        place_reward = 1 - torch.tanh(5.0 * cubeA_to_goal_dist)

        reward[info["is_cubeA_grasped"]] = (4 + place_reward)[info["is_cubeA_grasped"]]

        # ungrasp and static reward
        gripper_width = (self.agent.robot.get_qlimits()[0, -1, 1] * 2).to(
            self.device
        )  # NOTE: hard-coded with panda
        is_cubeA_grasped = info["is_cubeA_grasped"]
        ungrasp_reward = (
            torch.sum(self.agent.robot.get_qpos()[:, -2:], axis=1) / gripper_width
        )
        ungrasp_reward[~is_cubeA_grasped] = 1.0
        v = torch.linalg.norm(self.CubeA.linear_velocity, axis=1)
        av = torch.linalg.norm(self.CubeA.angular_velocity, axis=1)
        static_reward = 1 - torch.tanh(v * 10 + av)
        reward[info["is_cubeA_on_cubeB"]] = (
            6 + (ungrasp_reward + static_reward) / 2.0
        )[info["is_cubeA_on_cubeB"]]

        reward[info["success"]] = 8

        return reward
    
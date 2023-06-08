import logging
from typing import List

import attr
import numpy as np

from robogym.envs.rearrange.common.mesh import (
    MeshRearrangeEnv,
    MeshRearrangeEnvConstants,
    MeshRearrangeEnvParameters,
    MeshRearrangeSimParameters,
)
from robogym.envs.rearrange.common.utils import find_meshes_by_dirname
from robogym.envs.rearrange.goals.object_state_fixed import ObjectFixedStateGoal
from robogym.envs.rearrange.goals.task_reach_goal import TaskReachGoal
from robogym.envs.rearrange.simulation.base import ObjectGroupConfig
from robogym.envs.rearrange.simulation.mesh import MeshRearrangeSim
from robogym.envs.rearrange.ycb import find_ycb_meshes
from robogym.envs.rearrange.task_env import TaskEnv
from robogym.robot_env import build_nested_attr
from robogym.utils.rotation import quat_from_angle_and_axis
from robogym.envs.rearrange.common.utils import place_objects_with_no_constraint

logger = logging.getLogger(__name__)

class TaskReachEnv(TaskEnv):
    @classmethod
    def build_goal_generation(cls, constants, mujoco_simulation):
        #return an instance of the TaskReachGoal goal generator
        return TaskReachGoal(
            mujoco_simulation,
            args=constants.goal_args,
            )
    def _calculate_goal_distance_reward(self, previous_goal_distance, goal_distance):
        # Calculate goal distance reward by looking at how far the gripper's position in the current state is from 
        # the gripper's goal position
        return np.sum(previous_goal_distance["obj_pos"] - goal_distance["obj_pos"])
    
make_env = TaskReachEnv.build
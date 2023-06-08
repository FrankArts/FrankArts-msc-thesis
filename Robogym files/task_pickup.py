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
from robogym.envs.rearrange.goals.task_pickup_goal import TaskPickupGoal
from robogym.envs.rearrange.simulation.base import ObjectGroupConfig
from robogym.envs.rearrange.simulation.mesh import MeshRearrangeSim
from robogym.envs.rearrange.ycb import find_ycb_meshes
from robogym.envs.rearrange.task_env import TaskEnv
from robogym.robot_env import build_nested_attr
from robogym.utils.rotation import quat_from_angle_and_axis
from robogym.envs.rearrange.common.utils import place_objects_with_no_constraint

logger = logging.getLogger(__name__)

class TaskPickupEnv(TaskEnv):
    
    def _randomize_object_initial_positions(self):
        random_state = np.random.RandomState(seed = 0)
        positions, is_valid = self._generate_object_placements()
        self.goal_generation.initial_pos = positions.copy()
        print("_randomize_object_initial_positions:")
        print(positions)
        self.mujoco_simulation.set_object_pos(positions)
    
    @classmethod
    def build_goal_generation(cls, constants, mujoco_simulation):
        print("TaskPickupEnv.build_goal_generation()")
        return TaskPickupGoal(
            mujoco_simulation,
            args=constants.goal_args,
            )
    
make_env = TaskPickupEnv.build
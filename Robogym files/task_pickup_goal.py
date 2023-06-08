from typing import Dict, Tuple, Union

import numpy as np
from numpy.random import RandomState

from robogym.envs.rearrange.common.utils import place_objects_with_no_constraint
from robogym.envs.rearrange.goals.object_state import GoalArgs, ObjectStateGoal
from robogym.envs.rearrange.simulation.base import RearrangeSimulationInterface
from robogym.utils import rotation

class TaskPickupGoal(ObjectStateGoal):
    def __init__(
        self,
        mujoco_simulation: RearrangeSimulationInterface,
        args: Union[Dict, GoalArgs] = GoalArgs(),
    ):
        super().__init__(mujoco_simulation, args=args)
        self.mujoco_simulation: RearrangeSimulationInterface = mujoco_simulation
        
    def _sample_next_goal_positions(
        self, random_state: RandomState
    ) -> Tuple[np.ndarray, bool]:
        if hasattr(self, "initial_pos"):
            goal_positions = self.initial_pos[0:2]
            goal_valid = True
        else:
            goal_positions, goal_valid = place_objects_with_no_constraint(
                self.mujoco_simulation.get_object_bounding_boxes(),
                self.mujoco_simulation.get_table_dimensions(),
                self.mujoco_simulation.get_placement_area(),
                max_placement_trial_count=self.mujoco_simulation.max_placement_retry,
                max_placement_trial_count_per_object=self.mujoco_simulation.max_placement_retry_per_object,
                random_state=random_state,
            )
        
        #self.mujoco_simulation.set_object_pos(goal_positions)

        # Target object is located in index 0, so goal_position[0][:]
        # Increase the target object's goal height ([2]) by x
        goal_positions[0][2] += 0.1
        #self.mujoco_simulation.set_object_pos(goal_positions)
        print("goal_generation._sample_next_goal_positions():")
        print(goal_positions)
        #self.goal_positions = goal_positions
        return goal_positions, goal_valid
    
    def is_object_grasped(self):
        grasped = self.mujoco_simulation.get_object_gripper_contact()
        return np.array([x[0] + x[1] for x in grasped])
    
    def current_state(self) -> dict:
        """ Extract current goal state """
        gripper_pos = self.mujoco_simulation.mj_sim.data.get_site_xpos(
            "robot0:grip"
        ).copy()
        gripper_pos = np.array([gripper_pos])
        current_state = super().current_state()
        current_state.update(
            {"gripper_pos": gripper_pos, "grasped": self.is_object_grasped()}
        )
        return current_state
    
    def relative_goal(self, goal_state: dict, current_state: dict) -> dict:
        #How close is the gripper to grabbing an object?
        gripper_pos = current_state["obj_pos"] - current_state["gripper_pos"]
        #How close are the objects to their goal locations?
        obj_pos = goal_state["obj_pos"] - current_state["obj_pos"]
        
        return {
            "obj_pos": obj_pos,
            "gripper_pos": gripper_pos,
            "obj_rot": self.rot_dist_func(goal_state, current_state)
        }
    
    def goal_distance(self, goal_state: dict, current_state: dict) -> dict:
        relative_goal = self.relative_goal(goal_state, current_state)
        pos_distances = np.linalg.norm(relative_goal["obj_pos"], axis = -1)
        gripper_distances = np.linalg.norm(relative_goal["gripper_pos"], axis = -1)
        #Remove rot_distances (here and in the return) if we dont care about rotation
        rot_distances = rotation.quat_magnitude(
            rotation.quat_normalize(rotation.euler2quat(relative_goal["obj_rot"]))
        )
        
        return {
            "relative_goal": relative_goal.copy(),
            "gripper_pos": gripper_distances,
            "obj_pos": pos_distances,
            "obj_rot": rot_distances,
            "grasped": current_state["grasped"],
        }
    
    
    
    
    
    
    
    
    
    
    
    
    
    
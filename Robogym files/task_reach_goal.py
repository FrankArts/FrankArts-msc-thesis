from typing import Dict, Tuple, Union

import numpy as np
from numpy.random import RandomState

from robogym.envs.rearrange.common.utils import place_objects_with_no_constraint
from robogym.envs.rearrange.goals.object_state import GoalArgs, ObjectStateGoal
from robogym.envs.rearrange.simulation.base import RearrangeSimulationInterface
from robogym.utils import rotation

class TaskReachGoal(ObjectStateGoal):
    # Adapted from the BlocksReachGoal goal generator from the robogym library.
    # Differences are that the goal position is set on exactly the target object's initial position,
    #  instead of a set height above the position.
    # Additionally, this goal generator works with two objects, but only the position of the first (target) object
    # is considered. The second object functions as a distractor, and is ignored by this goal generator.
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
            obj_positions = self.initial_pos.copy()
            goal_valid = True
        else:
            obj_positions, goal_valid = place_objects_with_no_constraint(
                self.mujoco_simulation.get_object_bounding_boxes(),
                self.mujoco_simulation.get_table_dimensions(),
                self.mujoco_simulation.get_placement_area(),
                max_placement_trial_count=self.mujoco_simulation.max_placement_retry,
                max_placement_trial_count_per_object=self.mujoco_simulation.max_placement_retry_per_object,
                random_state=random_state,
            )
            self.initial_pos = obj_positions.copy()[0:2]
        goal_positions = obj_positions[0:2]
        return goal_positions, goal_valid
      
    def current_state(self) -> dict:
        # Returns the built-in current_state dictionary, but adds the current position of the gripper
        gripper_pos = self.mujoco_simulation.mj_sim.data.get_site_xpos(
            "robot0:grip"
        ).copy()
        gripper_pos = np.array(gripper_pos) 
        current_state = super().current_state()
        current_state.update(
            {"gripper_pos": gripper_pos}
        )
        return current_state
    
    def relative_goal(self, goal_state: dict, current_state: dict) -> dict:
        #How close are the objects to their goal locations?
        # "obj_pos" corresponds to how close the gripper is to the goal location.
        # "obj_rot" is unused, but removing it causes an error.
        gripper_pos = current_state["gripper_pos"]
        target_pos = self.initial_pos[0][0:3]
        relative_pos = target_pos - gripper_pos
        return {
            "obj_pos": relative_pos,
            "obj_rot": self.rot_dist_func(goal_state, current_state),
        }
    
    def goal_distance(self, goal_state: dict, current_state: dict) -> dict:
        # Distance from the current goal to the target state.
        # "obj_rot" is set to 0s as this value is not used, but removing it produces an error.
        gripper_pos = current_state["gripper_pos"]
        target_pos = self.initial_pos[0][0:3]
        relative_pos = target_pos - gripper_pos
        distance = np.linalg.norm(relative_pos, axis=-1)
        return {
            "relative_goal": self.relative_goal(goal_state, current_state).copy(),
            "obj_pos": distance, 
            "obj_rot": np.zeros_like(distance)
        }
    
    
    
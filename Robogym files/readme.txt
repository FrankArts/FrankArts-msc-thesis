Files used for the robogym environment.

TaskEnv contains the custom rearrange environment that handles the colors and shapes of the objects in the simulation. The reset function takes two color and shape values as parameters, and stores those parameters as local variables. The shape and color sampling functions are overwritten to use those stored local parameters to set the desired colors and shapes.
The TaskEnv class only takes care of the initialization, not the goal.

TaskReachEnv inherits from TaskEnv, so has the same initialization capabilities. TaskReachEnv adds a goal generator (TaskReachGoal) and goal distance reward calculator corresponding to the task where the robot arm tries to move its gripper to the position of the target object. 

TaskReachGoal is an adaptation from the existing ObjectReachGoal goal generator in robogym, ensuring that the goal is set to having the gripper position in the same spot as the target object's initial position. This goal generator also makes sure that the right state variables are returned to the rest of the robogym simulation.

TaskPickupEnv and TaskPickupGoal are an additional environment and goal_generator corresponding to the task where the robot arm tries to move the target object to a location directly above its initial location. However, these two classes did not end up being used in the thesis due to time limitations.
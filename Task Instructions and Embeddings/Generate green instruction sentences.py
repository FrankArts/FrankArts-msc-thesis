import numpy as np
import pickle

# Generate task instructions for the reach and pickup task,
#  using four descriptions of each task, color and shape,
#  for a total of 64 descriptions per unique task.
# This script specifically generates green and yellow objects instead of red and blue.

# NOTE: The yellow task instructions are not used in the final version of the thesis.

# Possible values and synonyms for each feature:
COLORS = {"green": ["green", "emerald", "olive", "lime"], "yellow": ["yellow", "amber", "gold", "lemon"]}
SHAPES = {"sphere": ["sphere", "orb", "marble", "ball"], "cube": ["cube", "box", "block", "dice"]}
ACTIONS = ["pickup", "reach"]

FILE_PATH = "C:/Users/Frank/Desktop/Tilburg University/Thesis/Task Instructions"

def generate_task_instructions(color_key, shape_key, action_key):
    instructions = []
    # Pick up instructions
    if action_key == "pickup":
        for color in COLORS[color_key]:
            for shape in SHAPES[shape_key]:        
                instructions += generate_pickup_instructions(color, shape)
    # Reach instructions
    else:
        for color in COLORS[color_key]:
            for shape in SHAPES[shape_key]:        
                instructions += generate_reach_instructions(color, shape)
    return instructions
def generate_pickup_instructions(color, shape):
    output = []
    output.append("Pick up the {} {}.".format(color, shape))
    output.append("Increase the {} {}'s elevation.".format(color, shape))
    output.append("Raise the {} {}.".format(color, shape))
    output.append("Move the {} {} higher.".format(color, shape))
    return output

def generate_reach_instructions(color, shape):
    output = []
    output.append("Point at the {} {}.".format(color, shape))
    output.append("Reach for the {} {}.".format(color, shape))
    output.append("Touch the {} {}.".format(color, shape))
    output.append("Make contact with the {} {}.".format(color, shape))
    return output

#print(generate_task_instructions("blue", "cube", "reach"))
all_tasks = []
for color in COLORS.keys():
    for shape in SHAPES.keys():
        for action in ACTIONS:
            all_tasks += generate_task_instructions(color, shape, action)
            
with open(FILE_PATH + "/task_instructions_green_yellow.pickle", 'wb') as handle:
    pickle.dump(all_tasks, handle, protocol=pickle.HIGHEST_PROTOCOL)


# Task instruction identifiers: 
    # 1, [Pickup, red, sphere], 0-63
    # 2, [Reach, red, sphere], 64-127
    # 3, [Pickup, red, cube], 128 - 191
    # 4, [Reach, red, cube], 192 - 255
    # 5, [Pickup, blue, sphere], 256 - 319
    # 6, [Reach, blue, sphere], 320 - 383
    # 7, [Pickup, blue, cube], 384 - 447
    # 8, [Reach, blue, cube], 448 - 511
task_identifiers = [1 + i//64 for i in range(len(all_tasks))]

with open(FILE_PATH + "/task_ids_green_yellow.pickle", 'wb') as handle:
    pickle.dump(task_identifiers, handle, protocol=pickle.HIGHEST_PROTOCOL)




























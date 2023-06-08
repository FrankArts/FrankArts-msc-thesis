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
from robogym.envs.rearrange.simulation.base import ObjectGroupConfig
from robogym.envs.rearrange.simulation.mesh import MeshRearrangeSim
from robogym.envs.rearrange.ycb import find_ycb_meshes
from robogym.robot_env import build_nested_attr
from robogym.utils.rotation import quat_from_angle_and_axis

logger = logging.getLogger(__name__)


@attr.s(auto_attribs=True)
class TaskEnvParameters(MeshRearrangeEnvParameters):
    simulation_params: MeshRearrangeSimParameters = build_nested_attr(
        MeshRearrangeSimParameters, default=dict(num_objects=2)
    )
    
class TaskEnv(
    MeshRearrangeEnv[
        TaskEnvParameters, MeshRearrangeEnvConstants, MeshRearrangeSim,
    ]
):
    MESH_FILES = find_meshes_by_dirname("geom")
    # def _build_observation_providers( self, *args, **kwargs ):
    #     #Change images from 3,200,200 to 3,50,50, speeding up processing by 16x
    #     self.constants.vision_args.image_size = 50
    #     return super()._build_observation_providers(*args, **kwargs)
    #     # Commented out, because this part is taken care of through the constants parameter
    #     # Keeping the comment here as backup in case the constants parameter does not work as expected
    
    def _sample_random_object_groups(
        self, dedupe_objects: bool = False
    ) -> List[ObjectGroupConfig]:
        return super()._sample_random_object_groups(dedupe_objects=True)

    def _sample_object_colors(self, num_groups: int):
	# Set the object colors according to the values stored in self.color1 and self.color2
        assert num_groups == 2
        red = [1.0, 0.0, 0.0, 1.0]
        green = [0.0, 1.0, 0.0, 1.0]
        blue = [0.0, 0.0, 1.0, 1.0]
        color1 = red if self.color1 == "red" else blue
        color2 = red if self.color2 == "red" else blue
        color1 = green if self.color1 == "green" else color1
        return [color1, color2]

    def _sample_object_size_scales(self, num_groups: int):
        assert num_groups == 2
        return [1, 1]

    def _sample_object_meshes(self, num_groups: int):
	# Set the shape of the objects according to the value stored in self.shape1 and self.shape2.
	# MESH_FILES['sphere80.stl'] refers a relatively simple sphere object.
	# Alternatively, keys 'sphere320.stl' and 'sphere1280.stl' are available for smoother spheres.
	# MESH_FILES['cube.stl'] refers to a simple cube object
        """Add 2 shapes."""
        sphere = self.MESH_FILES['sphere80.stl']
        cube = self.MESH_FILES['cube.stl']
        shape1 = sphere if self.shape1 == "sphere" else cube
        shape2 = sphere if self.shape2 == "sphere" else cube
        return [shape1, shape2]
    
    def reset(self, color1 = "red", shape1 = "cube", color2 = None, shape2 = None):
        # Set task features here: Color and shape for the main object, derive color/shape for secondary object, set task.
        # These values can then be used for making objects and making the goal.
        if color2 == None:
            color2 = "blue" if color1 == "red" else "red"
            if color1 == "green":
                color2 = np.random.choice(["red", "blue"])
        if shape2 == None:
            shape2 = "sphere" if shape1 == "cube" else "cube"
        
        self.color1 = color1
        self.color2 = color2
        self.shape1 = shape1
        self.shape2 = shape2
        return super().reset()

make_env = TaskEnv.build
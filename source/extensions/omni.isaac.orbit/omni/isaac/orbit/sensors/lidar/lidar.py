# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import numpy as np
import re
import torch
from collections.abc import Sequence
from tensordict import TensorDict
from typing import TYPE_CHECKING, Any, Literal
from rich import print

import omni.kit.commands
import omni.usd
from omni.isaac.core.prims import XFormPrimView

from pxr import UsdGeom

import omni.isaac.RangeSensorSchema as RangeSensorSchema
from omni.isaac.range_sensor import _range_sensor

import omni.isaac.orbit.sim as sim_utils
from omni.isaac.orbit.utils import to_camel_case
from omni.isaac.orbit.utils.array import convert_to_torch
from omni.isaac.orbit.utils.math import quat_from_matrix

from ..sensor_base import SensorBase
from .lidar_data import LidarData
from .utils import convert_orientation_convention, create_rotation_matrix_from_view

if TYPE_CHECKING:
    from .lidar_cfg import LidarCfg


class Lidar(SensorBase):
    r"""The camera sensor for acquiring visual data.

    This class wraps over the `UsdGeom Camera`_ for providing a consistent API for acquiring visual data.
    It ensures that the camera follows the ROS convention for the coordinate system.

    Summarizing from the `replicator extension`_, the following sensor types are supported:

    - ``"rgb"``: A rendered color image.
    - ``"distance_to_camera"``: An image containing the distance to camera optical center.
    - ``"distance_to_image_plane"``: An image containing distances of 3D points from camera plane along camera's z-axis.
    - ``"normals"``: An image containing the local surface normal vectors at each pixel.
    - ``"motion_vectors"``: An image containing the motion vector data at each pixel.
    - ``"instance_segmentation"``: The instance segmentation data.
    - ``"semantic_segmentation"``: The semantic segmentation data.

    .. note::
        Currently the following sensor types are not supported in a "view" format:

        - ``"bounding_box_2d_tight"``: The tight 2D bounding box data (only contains non-occluded regions).
        - ``"bounding_box_2d_loose"``: The loose 2D bounding box data (contains occluded regions).
        - ``"bounding_box_3d"``: The 3D view space bounding box data.

        In case you need to work with these sensor types, we recommend using the single camera implementation
        from the :mod:`omni.isaac.orbit.compat.camera` module.

    .. _replicator extension: https://docs.omniverse.nvidia.com/prod_extensions/prod_extensions/ext_replicator/annotators_details.html#annotator-output
    .. _USDGeom Camera: https://graphics.pixar.com/usd/docs/api/class_usd_geom_camera.html

    """

    cfg: LidarCfg
    """The configuration parameters."""
    # UNSUPPORTED_TYPES: set[str] = {"bounding_box_2d_tight", "bounding_box_2d_loose", "bounding_box_3d"}
    """The set of sensor types that are not supported by the camera class."""

    def __init__(self, cfg: LidarCfg):
        """Initializes the camera sensor.

        Args:
            cfg: The configuration parameters.

        Raises:
            RuntimeError: If no camera prim is found at the given path.
            ValueError: If the provided data types are not supported by the camera.
        """
        # check if sensor path is valid
        # note: currently we do not handle environment indices if there is a regex pattern in the leaf
        #   For example, if the prim path is "/World/Sensor_[1,2]".
        sensor_path = cfg.prim_path.split("/")[-1]
        sensor_path_is_regex = re.match(r"^[a-zA-Z0-9/_]+$", sensor_path) is None
        if sensor_path_is_regex:
            raise RuntimeError(
                f"Invalid prim path for the camera sensor: {self.cfg.prim_path}."
                "\n\tHint: Please ensure that the prim path does not contain any regex patterns in the leaf."
            )
        # perform check on supported data types
        # self._check_supported_data_types(cfg)
        # initialize base class
        super().__init__(cfg)

        # spawn the asset
        if self.cfg.spawn is not None:
            # compute the rotation offset
            rot = torch.tensor(self.cfg.offset.rot, dtype=torch.float32).unsqueeze(0)
            rot_offset = convert_orientation_convention(rot, origin=self.cfg.offset.convention, target="opengl")
            rot_offset = rot_offset.squeeze(0).numpy()
            # spawn the asset
            self.cfg.spawn.func(
                self.cfg.prim_path, self.cfg.spawn, translation=self.cfg.offset.pos, orientation=rot_offset
            )
        # check that spawn was successful
        matching_prims = sim_utils.find_matching_prims(self.cfg.prim_path)
        if len(matching_prims) == 0:
            raise RuntimeError(f"Could not find prim with path {self.cfg.prim_path}.")

        # UsdGeom Camera prim for the sensor
        self._sensor_prims = list()
        # Create empty variables for storing output data
        self._data = LidarData()

    def __del__(self):
        """Unsubscribes from callbacks and detach from the replicator registry."""
        # unsubscribe callbacks
        super().__del__()
        # delete from replicator registry
        for _, annotators in self._rep_registry.items():
            for annotator, render_product_path in zip(annotators, self._render_product_paths):
                annotator.detach([render_product_path])
                annotator = None

    def __str__(self) -> str:
        """Returns: A string containing information about the instance."""
        # message for class
        return (
            f"Camera @ '{self.cfg.prim_path}': \n"
            f"\tdata types   : {self.data.output.sorted_keys} \n"
            f"\tupdate period (s): {self.cfg.update_period}\n"
            # f"\tshape        : {self.image_shape}\n"
            f"\tnumber of sensors : {self._view.count}"
        )

    """
    Properties
    """

    @property
    def num_instances(self) -> int:
        return self._view.count

    @property
    def data(self) -> LidarData:
        # update sensors if needed
        self._update_outdated_buffers()
        # return the data
        return self._data

    @property
    def frame(self) -> torch.tensor:
        """Frame number when the measurement took place."""
        return self._frame

    @property
    def render_product_paths(self) -> list[str]:
        """The path of the render products for the cameras.

        This can be used via replicator interfaces to attach to writes or external annotator registry.
        """
        return self._render_product_paths

    @property
    def scan_properties(self) -> dict:
        """Provides essential properties of the LiDAR scan."""
        return {
            "num_points": self.cfg.num_rays,  # Number of points (rays) in each scan cycle
            "angular_range": self.cfg.angular_range,  # The angular range covered by the LiDAR scan
            "resolution": self.cfg.resolution,  # The angular resolution of the LiDAR scan
        }

    """
    Configuration
    """

    def set_lidar_properties(
        self,
        angular_range: float = 360.0,
        resolution: float = 1.0,
        max_distance: float = 100.0,
        env_ids: Sequence[int] | None = None
    ):
        """Set parameters of the USD LiDAR prim from the given specifications.

        Args:
            angular_range: The angular range of the LiDAR scan in degrees. Defaults to 360.0.
            resolution: The angular resolution of the scan in degrees. Defaults to 1.0.
            max_distance: The maximum distance that the LiDAR can measure. Defaults to 100.0.
            env_ids: A sequence of environment IDs to manipulate. Defaults to None, which means all sensor indices.
        """
        # Resolve environment IDs
        if env_ids is None:
            env_ids = self._ALL_INDICES

        # Iterate over environment IDs
        for i in env_ids:
            # Get corresponding sensor prim
            sensor_prim = self._sensor_prims[i]

            # Set LiDAR properties
            # Note: The following properties and their assignment to the USD prim are hypothetical
            # and need to be adapted to your specific simulation environment and LiDAR sensor representation.
            sensor_prim.GetAttribute("angularRange").Set(angular_range)
            sensor_prim.GetAttribute("resolution").Set(resolution)
            sensor_prim.GetAttribute("maxDistance").Set(max_distance)

            # Additional LiDAR-specific properties can be set here as needed.

    """
    Operations - Set pose.
    """

    def set_world_poses(
        self,
        positions: torch.Tensor | None = None,
        orientations: torch.Tensor | None = None,
        env_ids: Sequence[int] | None = None,
        convention: Literal["opengl", "ros", "world"] = "ros",
    ):
        r"""Set the pose of the camera w.r.t. the world frame using specified convention.

        Since different fields use different conventions for camera orientations, the method allows users to
        set the camera poses in the specified convention. Possible conventions are:

        - :obj:`"opengl"` - forward axis: -Z - up axis +Y - Offset is applied in the OpenGL (Usd.Camera) convention
        - :obj:`"ros"`    - forward axis: +Z - up axis -Y - Offset is applied in the ROS convention
        - :obj:`"world"`  - forward axis: +X - up axis +Z - Offset is applied in the World Frame convention

        See :meth:`omni.isaac.orbit.sensors.camera.utils.convert_orientation_convention` for more details
        on the conventions.

        Args:
            positions: The cartesian coordinates (in meters). Shape is (N, 3).
                Defaults to None, in which case the camera position in not changed.
            orientations: The quaternion orientation in (w, x, y, z). Shape is (N, 4).
                Defaults to None, in which case the camera orientation in not changed.
            env_ids: A sensor ids to manipulate. Defaults to None, which means all sensor indices.
            convention: The convention in which the poses are fed. Defaults to "ros".

        Raises:
            RuntimeError: If the camera prim is not set. Need to call :meth:`initialize` method first.
        """
        # resolve env_ids
        if env_ids is None:
            env_ids = self._ALL_INDICES
        # convert to backend tensor
        if positions is not None:
            if isinstance(positions, np.ndarray):
                positions = torch.from_numpy(positions).to(device=self._device)
            elif not isinstance(positions, torch.Tensor):
                positions = torch.tensor(positions, device=self._device)
        # convert rotation matrix from input convention to OpenGL
        if orientations is not None:
            if isinstance(orientations, np.ndarray):
                orientations = torch.from_numpy(orientations).to(device=self._device)
            elif not isinstance(orientations, torch.Tensor):
                orientations = torch.tensor(orientations, device=self._device)
            orientations = convert_orientation_convention(orientations, origin=convention, target="opengl")
        # set the pose
        self._view.set_world_poses(positions, orientations, env_ids)

    def set_world_poses_from_view(
        self, eyes: torch.Tensor, targets: torch.Tensor, env_ids: Sequence[int] | None = None
    ):
        """Set the poses of the camera from the eye position and look-at target position.

        Args:
            eyes: The positions of the camera's eye. Shape is (N, 3).
            targets: The target locations to look at. Shape is (N, 3).
            env_ids: A sensor ids to manipulate. Defaults to None, which means all sensor indices.

        Raises:
            RuntimeError: If the camera prim is not set. Need to call :meth:`initialize` method first.
            NotImplementedError: If the stage up-axis is not "Y" or "Z".
        """
        # resolve env_ids
        if env_ids is None:
            env_ids = self._ALL_INDICES
        # set camera poses using the view
        orientations = quat_from_matrix(create_rotation_matrix_from_view(eyes, targets, device=self._device))
        self._view.set_world_poses(eyes, orientations, env_ids)

    """
    Operations
    """

    def reset(self, env_ids: Sequence[int] | None = None):
        # reset the timestamps
        super().reset(env_ids)
        # resolve None
        # note: cannot do smart indexing here since we do a for loop over data.
        if env_ids is None:
            env_ids = self._ALL_INDICES
        # reset the data
        # note: this recomputation is useful if one performs randomization on the camera poses.
        self._update_poses(env_ids)
        # self._update_intrinsic_matrices(env_ids)
        # Reset the frame count
        self._frame[env_ids] = 0

    """
    Implementation.
    """

    def _initialize_impl(self):
        """Initializes the LiDAR sensor handles and internal buffers.

        This function prepares the LiDAR sensor for data collection, ensuring it is properly configured within the simulation environment. It also initializes the internal buffers to store the LiDAR data.

        Raises:
            RuntimeError: If the number of LiDAR prims in the view does not match the expected number.
        """
        import omni.replicator.core as rep

        # Initialize the base class
        super()._initialize_impl()

        # Prepare a view for the LiDAR sensor based on its path
        self._view = XFormPrimView(self.cfg.prim_path, reset_xform_properties=False)
        self._view.initialize()

        # Ensure the number of detected LiDAR prims matches the expected number
        if self._view.count != self._num_envs:
            raise RuntimeError(f"Expected number of LiDAR prims ({self._num_envs}) does not match the found number ({self._view.count}).")

        # Prepare environment ID buffers
        self._ALL_INDICES = torch.arange(self._view.count, device=self._device, dtype=torch.long)

        # Initialize a frame count buffer
        self._frame = torch.zeros(self._view.count, device=self._device, dtype=torch.long)
        self._render_product_paths: list[str] = list()
        self._rep_registry: dict[str, list[rep.annotators.Annotator]] = {name: list() for name in self.cfg.data_types}
        
        # Resolve device name
        if "cuda" in self._device:
            device_name = self._device.split(":")[0]
        else:
            device_name = "cpu"
        
        # NOTE: For LiDAR, the concept of "render products" might not apply in the same way as cameras, depending on your simulation setup.
        # If your LiDAR simulation outputs data that can be directly utilized (e.g., point clouds), you may not need to register with a replicator registry.
        # Below lines are placeholders for any LiDAR-specific initialization you may need.

        # Initialize any LiDAR-specific buffers or settings here.

        # For example, you might have a list to hold references to LiDAR sensor prims.
        self._sensor_prims = list()
        self._sensor_paths = list()

        # Search and validate LiDAR sensor prims in the simulation environment
        for lidar_prim_path in self._view.prim_paths:
            lidar_prim = omni.usd.get_context().get_stage().GetPrimAtPath(lidar_prim_path)
            
            # self.lidar = RangeSensorSchema.Lidar.Define(omni.usd.get_context().get_stage(),lidar_prim_path)
            # self.lidar.GetHorizontalFovAttr().Set(270)
            # self.lidar.GetHorizontalResolutionAttr().Set(0.25)

            # Ensure the prim is valid and represents a LiDAR sensor in your simulation setup.
            # This step may involve checking custom properties or metadata that identify the prim as a LiDAR sensor.
            if lidar_prim and self._is_valid_lidar_prim(lidar_prim):
                self._sensor_prims.append(lidar_prim)
                self._sensor_paths.append(lidar_prim_path)
            else:
                raise RuntimeError(f"Prim at path '{lidar_prim_path}' is not recognized as a valid LiDAR sensor.")

            render_prod_path = rep.create.render_product(lidar_prim_path, resolution=[1, 1])
            
            
            for name in self.cfg.data_types:
            
                rep_annotator = rep.AnnotatorRegistry.get_annotator(name, device=device_name)
                
                rep_annotator.attach(render_prod_path)
            
                self._rep_registry[name].append(rep_annotator)

        print(f"Replicator registry: {self._rep_registry}")
        # Create internal buffers for LiDAR data
        self._create_buffers()

    # TODO : Make this check
    def _is_valid_lidar_prim(self, prim):
        # Checking if a USD prim is a valid LiDAR sensor in simulation environment.
        return True


    def _update_buffers_impl(self, env_ids: Sequence[int]):
        # Increment frame count
        self._frame[env_ids] += 1
        # -- pose
        self._update_poses(env_ids)
        # -- read the data from annotator registry
        # check if buffer is called for the first time. If so then, allocate the memory
        if len(self._data.output.sorted_keys) == 0:
            # this is the first time buffer is called
            # it allocates memory for all the sensors
            self._create_annotator_data()
        else:
            # iterate over all the data types
            for name, annotators in self._rep_registry.items():
                # iterate over all the annotators
                for index in env_ids:
                    # get the output
                    output = annotators[index].get_data()
                    # process the output
                    data, info = self._process_annotator_output(output)
                    # add data to output
                    self._data.output[name][index] = data
                    # add info to output
                    self._data.info[index][name] = info

    """
    Private Helpers
    """

    def _check_supported_data_types(self, cfg: LidarCfg):
        """Checks if the data types are supported by the ray-caster camera."""
        # check if there is any intersection in unsupported types
        # reason: these use np structured data types which we can't yet convert to torch tensor
        common_elements = set(cfg.data_types) & Lidar.UNSUPPORTED_TYPES
        if common_elements:
            raise ValueError(
                f"Camera class does not support the following sensor types: {common_elements}."
                "\n\tThis is because these sensor types output numpy structured data types which"
                "can't be converted to torch tensors easily."
                "\n\tHint: If you need to work with these sensor types, we recommend using the single camera"
                " implementation from the omni.isaac.orbit.compat.camera module."
            )
            
    def _create_buffers(self):
        """Create buffers for storing LiDAR distance measurement data."""
        # Pose of the LiDAR sensors in the world
        self._data.pos_w = torch.zeros((self._view.count, 3), device=self._device)
        self._data.quat_w_world = torch.zeros((self._view.count, 4), device=self._device)

        # Preparing a buffer for distance measurements. Assuming each LiDAR scan produces a fixed number of measurements,
        # the shape of the distance measurements buffer could be [number_of_sensors, number_of_measurements_per_scan].
        # The exact shape and initialization will depend on your specific sensor configuration and scanning pattern.
        self._data.distance_measurements = torch.zeros((self._view.count, self.cfg.num_rays), device=self._device)

        # Metadata about each scan, such as the scan sequence number or timestamp, could be useful for analysis or debugging.
        # This information structure is placeholder and can be adapted to your requirements.
        self._data.info = [{"scan_id": None, "timestamp": None} for _ in range(self._view.count)]
        self._data.output = TensorDict({}, batch_size=self._view.count, device=self.device)
        # Note: The 'output' buffer is not explicitly created here, as 'distance_measurements' effectively serves that purpose.
        # If additional output types or formats are required, consider adding them similarly.



    def _update_poses(self, env_ids: Sequence[int]):
        """Computes the pose of the camera in the world frame with ROS convention.

        This methods uses the ROS convention to resolve the input pose. In this convention,
        we assume that the camera front-axis is +Z-axis and up-axis is -Y-axis.

        Returns:
            A tuple of the position (in meters) and quaternion (w, x, y, z).
        """
        # check camera prim exists
        if len(self._sensor_prims) == 0:
            raise RuntimeError("Camera prim is None. Please call 'sim.play()' first.")

        # get the poses from the view
        poses, quat = self._view.get_world_poses(env_ids)
        self._data.pos_w[env_ids] = poses
        self._data.quat_w_world[env_ids] = convert_orientation_convention(quat, origin="opengl", target="world")

    def _create_annotator_data(self):
        """Create the buffers to store the annotator data.

        We create a buffer for each annotator and store the data in a dictionary. Since the data
        shape is not known beforehand, we create a list of buffers and concatenate them later.

        This is an expensive operation and should be called only once.
        """
        # add data from the annotators
        for name, annotators in self._rep_registry.items():
            # create a list to store the data for each annotator
            data_all_cameras = list()
            # iterate over all the annotators
            
            for index in self._ALL_INDICES:
                # get the output
                
                # Debugging prints
                print(annotators[0].get_name())
                #len
                # print(len(annotators[index].get_data()))
                
                output = annotators[index].get_data()
                # process the output
                data, info = self._process_annotator_output(output)
                # append the data
                data_all_cameras.append(data)
                # store the info
                self._data.info[index][name] = info
            # concatenate the data along the batch dimension
            self._data.output[name] = torch.stack(data_all_cameras, dim=0)

    def _process_annotator_output(self, output: Any) -> tuple[torch.tensor, dict]:
        """Process the annotator output.

        This function is called after the data has been collected from all the cameras.
        """
        # print(output)
        self._li = _range_sensor.acquire_lidar_sensor_interface()
        depth = self._li.get_depth_data(self._sensor_paths[0])
        zenith = self._li.get_zenith_data(self._sensor_paths[0])
        azimuth = self._li.get_azimuth_data(self._sensor_paths[0])
        linear_depth = self._li.get_linear_depth_data(self._sensor_paths[0])
        intensities = self._li.get_intensity_data(self._sensor_paths[0])
        num_cols = self._li.get_num_cols(self._sensor_paths[0])
        num_rows = self._li.get_num_rows(self._sensor_paths[0])
        # exec_out = self._li.get(self._sensor_paths[0])
        point_cloud = self._li.get_point_cloud_data(self._sensor_paths[0])
        print(f"point_cloud: {point_cloud}, point_cloud: {point_cloud.shape}, len(point_cloud): {len(point_cloud)}")
        # print(f"exec_out: {linear_depth}, linear_depth: {linear_depth.shape}")
        
        # extract info and data from the output
        if isinstance(output, dict):
            data = output["data"]
            info = output["info"]
            # print(f"info: {info}")
        else:
            data = output
            info = None
        # convert data into torch tensor
        data = convert_to_torch(data, device=self.device)
        # return the data and info
        return data, info

    """
    Internal simulation callbacks.
    """

    def _invalidate_initialize_callback(self, event):
        """Invalidates the scene elements."""
        # call parent
        super()._invalidate_initialize_callback(event)
        # set all existing views to None to invalidate them
        self._view = None


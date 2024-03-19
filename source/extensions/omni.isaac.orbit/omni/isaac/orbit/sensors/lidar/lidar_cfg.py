# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING
from typing import Literal

from omni.isaac.orbit.sim import FisheyeCameraCfg, PinholeCameraCfg
from omni.isaac.orbit.utils import configclass

from ..sensor_base_cfg import SensorBaseCfg
from .lidar import Lidar


@configclass
class LidarCfg(SensorBaseCfg):
    """Configuration for a camera sensor."""

    @configclass
    class OffsetCfg:
        """The offset pose of the sensor's frame from the sensor's parent frame."""

        pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
        """Translation w.r.t. the parent frame. Defaults to (0.0, 0.0, 0.0)."""

        rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
        """Quaternion rotation (w, x, y, z) w.r.t. the parent frame. Defaults to (1.0, 0.0, 0.0, 0.0)."""

        convention: Literal["opengl", "ros", "world"] = "ros"
        """The convention in which the frame offset is applied. Defaults to "ros".

        - ``"opengl"`` - forward axis: ``-Z`` - up axis: ``+Y`` - Offset is applied in the OpenGL (Usd.Camera) convention.
        - ``"ros"``    - forward axis: ``+Z`` - up axis: ``-Y`` - Offset is applied in the ROS convention.
        - ``"world"``  - forward axis: ``+X`` - up axis: ``+Z`` - Offset is applied in the World Frame convention.

        """

    class_type: type = Lidar

    offset: OffsetCfg = OffsetCfg()
    """The offset pose of the sensor's frame from the sensor's parent frame. Defaults to identity.

    Note:
        The parent frame is the frame the sensor attaches to. For example, the parent frame of a
        camera at path ``/World/envs/env_0/Robot/Camera`` is ``/World/envs/env_0/Robot``.
    """
    
    data_types: list[str] = ["pointcloud"]
    
    """
    No annotator of name ranges registered. Available annotators: ['camera_params', 'rgb', 'normals', 'motion_vectors', 'cross_correspondence', 'occlusion', 'distance_to_image_plane', 'distance_to_camera', 'LdrColor', 'HdrColor', 'SmoothNormal', 'BumpNormal', 'AmbientOcclusion', 'Motion2d', 'DiffuseAlbedo', 'SpecularAlbedo', 'Roughness', 'DirectDiffuse', 'DirectSpecular', 'Reflections', 'IndirectDiffuse', 'DepthLinearized', 'EmissionAndForegroundMask', 'PtDirectIllumation', 'PtGlobalIllumination', 'PtReflections', 'PtRefractions', 'PtSelfIllumination', 'PtBackground', 'PtWorldNormal', 'PtWorldPos', 'PtZDepth', 'PtVolumes', 'PtDiffuseFilter', 'PtReflectionFilter', 'PtRefractionFilter', 'PtMultiMatte0', 'PtMultiMatte1', 'PtMultiMatte2', 'PtMultiMatte3', 'PtMultiMatte4', 'PtMultiMatte5', 'PtMultiMatte6', 'PtMultiMatte7', 'primPaths', 'bounding_box_2d_tight_fast', 'bounding_box_2d_tight', 'bounding_box_2d_loose_fast', 'bounding_box_2d_loose', 'bounding_box_3d_360', 'bounding_box_3d_fast', 'bounding_box_3d', 'semantic_segmentation', 'instance_segmentation_fast', 'instance_segmentation', 'CameraParams', 'skeleton_data', 'pointcloud', 'CrossCorrespondence', 'MotionVectors', 'IsaacNoop', 'IsaacReadCameraInfo', 'IsaacReadTimes', 'IsaacReadSimulationTime', 'LdrColorSDIsaacConvertRGBAToRGB', 'DistanceToImagePlaneSDIsaacConvertDepthToPointCloud', 'RtxSensorCpuIsaacReadRTXLidarData', 'RtxSensorCpuIsaacComputeRTXLidarPointCloud', 'RtxSensorCpuIsaacCreateRTXLidarScanBuffer', 'RtxSensorCpuIsaacComputeRTXLidarFlatScan', 'RtxSensorCpuIsaacComputeRTXRadarPointCloud']
    """

    spawn: PinholeCameraCfg | FisheyeCameraCfg | None = MISSING
    """Spawn configuration for the asset.

    If None, then the prim is not spawned by the asset. Instead, it is assumed that the
    asset is already present in the scene.
    """

    scanning_range: tuple[float, float] = (0.1, 100.0)
    """Minimum and maximum scanning distance (in meters). Defaults to (0.1, 100.0)."""

    angular_range: float = 360.0
    """Horizontal field of view (in degrees). Defaults to 360.0 for a full circular scan."""

    vertical_fov: float = 30.0
    """Vertical field of view (in degrees). Defaults to 30.0, which might represent a flat 2D scan."""

    resolution: float = 0.1
    """Angular resolution (in degrees). Determines the spacing between individual rays in the scan."""

    num_rays: int = 1081
    """Number of rays used in the scan. This determines the LiDAR's resolution."""

    update_period: float = 0.025
    """Time between sensor updates (in seconds). For a 40Hz LiDAR, this would be 0.025."""

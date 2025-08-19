#!/usr/bin/env python3
"""
Camera System Architecture
Milestone 5: Flexible perception foundation for DINO integration
"""

import numpy as np
import pybullet as p
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from interfaces import SimulationComponent


@dataclass
class CameraConfiguration:
    """Camera system configuration parameters"""
    image_width: int = 640
    image_height: int = 480
    fov: float = 60.0  # Field of view in degrees
    near_plane: float = 0.1
    far_plane: float = 10.0


@dataclass
class RGBDData:
    """RGB-D camera capture data"""
    rgb_image: np.ndarray
    depth_image: np.ndarray
    camera_position: Tuple[float, float, float]
    camera_target: Tuple[float, float, float]
    timestamp: float


class Camera:
    """Individual camera with capture capabilities"""
    
    def __init__(self, position: Tuple[float, float, float], 
                 target: Tuple[float, float, float],
                 config: CameraConfiguration):
        self.position = position
        self.target = target
        self.config = config
        self.view_matrix = None
        self.projection_matrix = None
        self._update_matrices()
        
    def _update_matrices(self) -> None:
        """Update view and projection matrices"""
        # View matrix
        self.view_matrix = p.computeViewMatrix(
            cameraEyePosition=self.position,
            cameraTargetPosition=self.target,
            cameraUpVector=[0, 0, 1]
        )
        
        # Projection matrix
        aspect = self.config.image_width / self.config.image_height
        self.projection_matrix = p.computeProjectionMatrixFOV(
            fov=self.config.fov,
            aspect=aspect,
            nearVal=self.config.near_plane,
            farVal=self.config.far_plane
        )
        
    def capture_rgbd(self) -> RGBDData:
        """Capture RGB-D image"""
        # Capture image using PyBullet
        width, height, rgb_img, depth_img, _ = p.getCameraImage(
            width=self.config.image_width,
            height=self.config.image_height,
            viewMatrix=self.view_matrix,
            projectionMatrix=self.projection_matrix
        )
        
        # Convert RGB to numpy array (remove alpha channel)
        rgb_array = np.array(rgb_img).reshape(height, width, 4)[:, :, :3]
        
        # Convert depth to numpy array and normalize
        depth_array = np.array(depth_img).reshape(height, width)
        far = self.config.far_plane
        near = self.config.near_plane
        depth_normalized = far * near / (far - (far - near) * depth_array)
        
        import time
        return RGBDData(
            rgb_image=rgb_array,
            depth_image=depth_normalized,
            camera_position=self.position,
            camera_target=self.target,
            timestamp=time.time()
        )


class CameraSystem(SimulationComponent):
    """Flexible camera system supporting multiple configurations"""
    
    def __init__(self, config: CameraConfiguration):
        self.config = config
        self.cameras: Dict[str, Camera] = {}
        self.physics_client = None
        
    def initialize(self, use_gui: bool = False) -> None:
        """Initialize camera system"""
        # Camera system doesn't need physics client setup
        pass
        
    def setup_front_facing_camera(self, robot_height: float = 0.8) -> Camera:
        """Primary front-facing camera at robot height"""
        # Position camera in front of shelf area
        camera_position = (1.0, 0.0, robot_height)  # 1m in front of shelf
        camera_target = (0.0, 0.0, robot_height * 0.7)  # Look at shelf area
        
        camera = Camera(camera_position, camera_target, self.config)
        self.cameras["front"] = camera
        return camera
        
    def setup_top_down_camera(self, shelf_center: Tuple[float, float, float]) -> Camera:
        """Optional top-down camera for shelf"""
        x, y, z = shelf_center
        camera_position = (x, y, z + 1.0)  # 1m above shelf
        camera_target = (x, y, z)  # Look down at shelf
        
        camera = Camera(camera_position, camera_target, self.config)
        camera_name = f"top_down_{len([k for k in self.cameras.keys() if k.startswith('top_down')])}"
        self.cameras[camera_name] = camera
        return camera
        
    def capture_all_cameras(self) -> Dict[str, RGBDData]:
        """Capture RGB-D data from all cameras"""
        captures = {}
        for name, camera in self.cameras.items():
            captures[name] = camera.capture_rgbd()
        return captures
        
    def get_camera(self, name: str) -> Optional[Camera]:
        """Get camera by name"""
        return self.cameras.get(name)
        
    def cleanup(self) -> None:
        """Clean up camera resources"""
        self.cameras.clear()
        
    def get_state(self) -> dict:
        """Get current camera system state"""
        return {
            "num_cameras": len(self.cameras),
            "camera_names": list(self.cameras.keys()),
            "config": {
                "width": self.config.image_width,
                "height": self.config.image_height,
                "fov": self.config.fov
            }
        }
"""
Coordinate Transforms for Robot Grasping System

Handles pixel-to-world coordinate transformation with proper camera matrix handling.
Fixes the double matrix inversion bug identified in debugging.
"""

import numpy as np
import pybullet as p
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class CoordinateTransformer:
    """Handles coordinate transformations between pixel and world coordinates."""
    
    def __init__(self):
        self.camera_intrinsics: Optional[np.ndarray] = None
        self.camera_extrinsics: Optional[np.ndarray] = None
        self.image_width: int = 640
        self.image_height: int = 480
        
    def set_camera_parameters(self, 
                            intrinsics: np.ndarray,
                            view_matrix: np.ndarray,
                            width: int = 640, 
                            height: int = 480) -> None:
        """
        Set camera parameters for coordinate transformation.
        
        Args:
            intrinsics: 3x3 camera intrinsic matrix
            view_matrix: 4x4 PyBullet view matrix (world-to-camera transform)
            width: Image width in pixels
            height: Image height in pixels
        """
        self.camera_intrinsics = intrinsics.copy()
        # FIXED: PyBullet view matrix is already world-to-camera transform
        # Store as world-to-camera for direct use (no inversion needed)
        self.camera_extrinsics = view_matrix.copy()
        self.image_width = width
        self.image_height = height
        
        logger.info(f"Camera parameters set: {width}x{height}")
        logger.debug(f"Intrinsics:\n{self.camera_intrinsics}")
        
    def pixel_to_world_coordinates(self, u: float, v: float, depth: float) -> np.ndarray:
        """
        Convert pixel coordinates to world coordinates.
        
        Args:
            u: Pixel x coordinate
            v: Pixel y coordinate  
            depth: Depth value in meters
            
        Returns:
            3D world coordinates as numpy array
        """
        if self.camera_intrinsics is None or self.camera_extrinsics is None:
            raise ValueError("Camera parameters not set")
            
        if depth <= 0:
            raise ValueError(f"Invalid depth value: {depth}")
            
        # Convert pixel to camera coordinates
        fx, fy = self.camera_intrinsics[0, 0], self.camera_intrinsics[1, 1]
        cx, cy = self.camera_intrinsics[0, 2], self.camera_intrinsics[1, 2]
        
        # Camera coordinates (OpenGL convention: Z negative forward)
        x_cam = (u - cx) * depth / fx
        y_cam = (v - cy) * depth / fy
        z_cam = -depth  # Negative Z for OpenGL coordinate system
        
        # Homogeneous camera coordinates
        camera_point = np.array([x_cam, y_cam, z_cam, 1.0])
        
        # Convert to world coordinates using camera-to-world transform
        # FIXED: Invert world-to-camera to get camera-to-world
        camera_to_world = np.linalg.inv(self.camera_extrinsics)
        world_point = camera_to_world @ camera_point
        
        return world_point[:3]
        
    def world_to_pixel_coordinates(self, world_point: np.ndarray) -> Tuple[int, int, float]:
        """
        Convert world coordinates to pixel coordinates.
        
        Args:
            world_point: 3D world coordinates
            
        Returns:
            Tuple of (u, v, depth) pixel coordinates and depth
        """
        if self.camera_intrinsics is None or self.camera_extrinsics is None:
            raise ValueError("Camera parameters not set")
            
        # Convert to homogeneous coordinates
        world_homo = np.append(world_point, 1.0)
        
        # Transform to camera coordinates using world-to-camera transform
        camera_point = self.camera_extrinsics @ world_homo
        
        if camera_point[2] >= 0:  # In OpenGL, Z negative means in front
            raise ValueError("Point behind camera")
            
        # Project to pixel coordinates
        fx, fy = self.camera_intrinsics[0, 0], self.camera_intrinsics[1, 1]
        cx, cy = self.camera_intrinsics[0, 2], self.camera_intrinsics[1, 2]
        
        u = int(camera_point[0] * fx / camera_point[2] + cx)
        v = int(camera_point[1] * fy / camera_point[2] + cy)
        depth = -camera_point[2]  # Return positive depth distance
        
        return u, v, depth


class VisionSystemIntegrator:
    """Integrates with existing vision system for camera parameters."""
    
    def __init__(self):
        self.transformer = CoordinateTransformer()
        
    def setup_from_camera_system(self, 
                                camera_pos: np.ndarray,
                                camera_target: np.ndarray,
                                camera_up: np.ndarray,
                                fov: float = 60.0,
                                aspect: float = 4/3,
                                near: float = 0.1,
                                far: float = 10.0,
                                width: int = 640,
                                height: int = 480) -> CoordinateTransformer:
        """
        Setup coordinate transformer from camera system parameters.
        
        Args:
            camera_pos: Camera position in world coordinates
            camera_target: Camera target point in world coordinates
            camera_up: Camera up vector
            fov: Field of view in degrees
            aspect: Aspect ratio
            near: Near clipping plane
            far: Far clipping plane
            width: Image width
            height: Image height
            
        Returns:
            Configured coordinate transformer
        """
        # Create PyBullet view matrix - needs to be transposed!
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=camera_pos,
            cameraTargetPosition=camera_target,
            cameraUpVector=camera_up
        )
        # PyBullet returns column-major, need to transpose to row-major
        view_matrix = np.array(view_matrix).reshape(4, 4).T
        
        # Create projection matrix
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=fov,
            aspect=aspect,
            nearVal=near,
            farVal=far
        )
        proj_matrix = np.array(proj_matrix).reshape(4, 4)
        
        # Calculate intrinsics correctly - for square pixels, fx = fy
        # Compute focal length from vertical FOV and image height
        f = height / (2.0 * np.tan(np.radians(fov) / 2.0))
        fx = f  # Same focal length for both axes (square pixels)
        fy = f
        cx = width / 2.0
        cy = height / 2.0
        
        intrinsics = np.array([
            [fx, 0,  cx],
            [0,  fy, cy], 
            [0,  0,  1]
        ])
        
        # Set parameters in transformer
        self.transformer.set_camera_parameters(
            intrinsics=intrinsics,
            view_matrix=view_matrix,
            width=width,
            height=height
        )
        
        logger.info("Vision system integration complete")
        return self.transformer
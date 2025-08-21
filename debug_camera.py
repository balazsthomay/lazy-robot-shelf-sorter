#!/usr/bin/env python3
"""
Debug camera rendering - simplified test to isolate camera issues
"""
import sys
sys.path.append('src')

import pybullet as p
import pybullet_data
import numpy as np

def debug_camera_rendering():
    """Test basic camera rendering with minimal scene"""
    print("🔬 Debug: Basic camera rendering test")
    
    # Connect to physics
    physics_client = p.connect(p.GUI)
    p.setGravity(0, 0, -9.81)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    
    try:
        # Load ground plane
        p.loadURDF("plane.urdf")
        
        # Create a simple, large, high-contrast object
        object_id = p.loadURDF(
            "cube.urdf",
            [0.4, -0.2, 0.1],  # Raised higher, larger scale
            globalScaling=0.2   # Much larger: 20cm instead of 5cm
        )
        
        # Add a few physics steps
        for _ in range(10):
            p.stepSimulation()
        
        # Test multiple camera configurations
        configurations = [
            {"pos": (0.4, -0.2, 1.0), "target": (0.4, -0.2, 0.1), "desc": "Direct overhead"},
            {"pos": (0.6, -0.4, 0.8), "target": (0.4, -0.2, 0.1), "desc": "Angled view"},
            {"pos": (0.4, -0.2, 0.5), "target": (0.4, -0.2, 0.1), "desc": "Lower overhead"}
        ]
        
        for i, config in enumerate(configurations):
            print(f"\n🎯 Test {i+1}: {config['desc']}")
            print(f"   Camera: {config['pos']} -> {config['target']}")
            
            # Create view and projection matrices
            view_matrix = p.computeViewMatrix(
                cameraEyePosition=config["pos"],
                cameraTargetPosition=config["target"],
                cameraUpVector=[0, 0, 1]
            )
            
            projection_matrix = p.computeProjectionMatrixFOV(
                fov=60.0,
                aspect=640/480,
                nearVal=0.1,
                farVal=10.0
            )
            
            # Capture image
            width, height, rgb_img, depth_img, _ = p.getCameraImage(
                width=640,
                height=480,
                viewMatrix=view_matrix,
                projectionMatrix=projection_matrix,
                physicsClientId=physics_client
            )
            
            # Process depth
            depth_array = np.array(depth_img).reshape(height, width)
            
            # Analyze depth values
            print(f"   Depth stats: min={depth_array.min():.3f}, max={depth_array.max():.3f}, mean={depth_array.mean():.3f}")
            print(f"   Center pixel depth: {depth_array[240, 320]:.3f}")
            
            # Check for variation in depth
            unique_depths = len(np.unique(np.round(depth_array, 3)))
            print(f"   Unique depth values: {unique_depths}")
            
            # Check if we have any close pixels (object detection)
            close_pixels = np.sum(depth_array < 0.9)  # Less than 90cm
            print(f"   Close pixels (<0.9m): {close_pixels} / {width*height}")
            
    finally:
        p.disconnect()

if __name__ == "__main__":
    debug_camera_rendering()
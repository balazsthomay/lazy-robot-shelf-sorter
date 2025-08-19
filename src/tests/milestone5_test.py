#!/usr/bin/env python3
"""
Milestone 5 Test - Camera System Architecture
Part of Phase 1: Foundation - Milestone 5
"""

import time
import numpy as np
import pybullet as p
import pybullet_data
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vision import CameraSystem, CameraConfiguration
from simulation import ShelfEnvironment, ShelfConfiguration
from objects import ObjectLibrary


def test_camera_system() -> bool:
    """Test camera system functionality"""
    print("🚀 Testing camera system...")
    
    # Initialize PyBullet
    physics_client = p.connect(p.DIRECT)
    p.setGravity(0, 0, -9.81)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF("plane.urdf")
    
    # Create environment
    config = ShelfConfiguration(num_shelves=2)
    env = ShelfEnvironment(config)
    env.initialize(use_gui=False)
    shelf_ids = env.create_shelves()
    shelf_positions = env.get_shelf_positions()
    
    # Initialize camera system
    camera_config = CameraConfiguration(
        image_width=320,  # Smaller for faster testing
        image_height=240,
        fov=60.0
    )
    camera_system = CameraSystem(camera_config)
    
    # Setup cameras
    front_camera = camera_system.setup_front_facing_camera(robot_height=0.8)
    top_camera = camera_system.setup_top_down_camera(shelf_positions[0])
    
    print(f"  Created {len(camera_system.cameras)} cameras")
    
    # Test image capture
    start_time = time.time()
    captures = camera_system.capture_all_cameras()
    capture_time = time.time() - start_time
    
    print(f"  Captured images in {capture_time:.3f}s")
    
    # Validate captures
    success = True
    for name, rgbd_data in captures.items():
        rgb_shape = rgbd_data.rgb_image.shape
        depth_shape = rgbd_data.depth_image.shape
        
        expected_rgb = (240, 320, 3)  # H, W, C
        expected_depth = (240, 320)   # H, W
        
        if rgb_shape != expected_rgb:
            print(f"    ❌ {name} RGB shape: {rgb_shape}, expected {expected_rgb}")
            success = False
        
        if depth_shape != expected_depth:
            print(f"    ❌ {name} depth shape: {depth_shape}, expected {expected_depth}")
            success = False
            
        # Check for valid image data
        if np.all(rgbd_data.rgb_image == 0):
            print(f"    ❌ {name} RGB image is all black")
            success = False
            
        print(f"    ✅ {name} camera: RGB {rgb_shape}, Depth {depth_shape}")
    
    # Cleanup
    camera_system.cleanup()
    env.cleanup()
    p.disconnect()
    
    return success


def test_camera_with_objects() -> bool:
    """Test camera integration with object library"""
    print("🚀 Testing camera with objects...")
    
    # Initialize PyBullet
    physics_client = p.connect(p.DIRECT)
    p.setGravity(0, 0, -9.81)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF("plane.urdf")
    
    # Create environment with objects
    config = ShelfConfiguration(num_shelves=1)
    env = ShelfEnvironment(config)
    env.initialize(use_gui=False)
    shelf_ids = env.create_shelves()
    shelf_positions = env.get_shelf_positions()
    
    # Load some objects
    library = ObjectLibrary("assets")
    library.scan_objects()
    set_5, _, _ = library.get_progressive_sets()
    library.load_objects(set_5[:3], physics_client)  # Load 3 objects
    
    # Setup camera
    camera_config = CameraConfiguration()
    camera_system = CameraSystem(camera_config)
    front_camera = camera_system.setup_front_facing_camera()
    
    # Run physics simulation
    for _ in range(100):
        p.stepSimulation()
    
    # Capture scene with objects
    start_time = time.time()
    rgbd_data = front_camera.capture_rgbd()
    capture_time = time.time() - start_time
    
    # Analyze captured image
    rgb_mean = np.mean(rgbd_data.rgb_image)
    depth_mean = np.mean(rgbd_data.depth_image)
    
    print(f"  Capture time: {capture_time:.3f}s")
    print(f"  RGB mean value: {rgb_mean:.1f}")
    print(f"  Depth mean value: {depth_mean:.3f}m")
    
    # Success if we captured non-empty images
    success = (rgb_mean > 10 and  # Not completely black
               depth_mean > 0.1 and  # Valid depth values
               capture_time < 0.5)    # Reasonable performance
    
    # Cleanup
    library.unload_all()
    camera_system.cleanup()
    env.cleanup()
    p.disconnect()
    
    return success


def main():
    """Run Milestone 5 tests"""
    print("🚀 MILESTONE 5: CAMERA SYSTEM ARCHITECTURE")
    print("=" * 50)
    
    tests = [
        ("Camera System Functionality", test_camera_system),
        ("Camera-Object Integration", test_camera_with_objects),
    ]
    
    results = {}
    total_start = time.time()
    
    for test_name, test_func in tests:
        print(f"\n{test_name}...")
        start_time = time.time()
        
        try:
            result = test_func()
            results[test_name] = result
        except Exception as e:
            print(f"  ❌ Test failed with exception: {e}")
            results[test_name] = False
            
        test_time = time.time() - start_time
        print(f"  Time: {test_time:.3f}s")
        
    total_time = time.time() - total_start
    
    # Final results
    print("\n" + "=" * 50)
    print("📊 MILESTONE 5 - FINAL RESULTS")
    print("=" * 50)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        
    print(f"Total test time: {total_time:.3f}s")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n🎉 MILESTONE 5 COMPLETE!")
        print("✅ Camera System Architecture ready")
        print("\nCamera System Features Validated:")
        print("- Front-facing camera at robot height")
        print("- Top-down camera for shelf monitoring")
        print("- RGB-D image capture functionality")
        print("- Integration with object library")
        print("- Flexible camera configuration")
    else:
        print("\n❌ Milestone 5 validation failed")
        print("Camera system needs attention")
        
    return all_passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
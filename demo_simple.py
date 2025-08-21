#!/usr/bin/env python3
"""
Phase 4.5 Robot Control - Vision-Guided Demo
Uses object detection to automatically find objects instead of hardcoded coordinates
"""

import sys
import os
sys.path.append('src')

import pybullet as p
import pybullet_data
import time
from control import RobotController, MotionController
from placement import PlacementCommand
from vision import CameraSystem, CameraConfiguration
from object_detection import ObjectDetector


def main():
    """Vision-guided demonstration with automatic object detection"""
    print("🚀 Phase 4.5 - Vision-Guided Demo")  
    print("="*50)
    print("🎯 Using automatic object detection - no manual coordinates!")
    
    # Connect to PyBullet GUI
    physics_client = p.connect(p.GUI)
    p.setGravity(0, 0, -9.81)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    
    # Physics optimization for stable grasping
    p.setTimeStep(1./1000.)  # Smaller timestep (1ms) for stability
    p.setPhysicsEngineParameter(numSolverIterations=150)  # More solver iterations
    p.setPhysicsEngineParameter(numSubSteps=80)  # More substeps
    
    # Better camera view
    p.resetDebugVisualizerCamera(
        cameraDistance=1.5,
        cameraYaw=30,
        cameraPitch=-30,
        cameraTargetPosition=[0, -0.5, 0.3]
    )
    
    try:
        # Load ground
        p.loadURDF("plane.urdf")
        
        # Create simple shelf - FAR FROM ROBOT AND PICKUP
        shelf_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=p.createCollisionShape(
                p.GEOM_BOX, halfExtents=[0.3, 0.2, 0.01]
            ),
            baseVisualShapeIndex=p.createVisualShape(
                p.GEOM_BOX, halfExtents=[0.3, 0.2, 0.01], 
                rgbaColor=[0.8, 0.6, 0.4, 1.0]
            ),
            basePosition=[0.7, 0.3, 0.4]  # MUCH FARTHER: opposite side, higher
        )
        
        # Add object to pick up - ROBOT'S PRECISE REACH ZONE
        object_id = p.loadURDF(
            "cube.urdf",
            [0.4, -0.2, 0.04],  # Lower position for better grasping
            globalScaling=0.05  # Smaller for better gripper fit (5cm cube)
        )
        
        # Add friction to the cube for better grasping
        p.changeDynamics(object_id, -1, 
                        lateralFriction=1.0, 
                        spinningFriction=0.005,
                        rollingFriction=0.005)
        
        # Visual markers - UPDATED FOR PRECISE REACH
        p.addUserDebugText("PICKUP OBJECT", [0.4, -0.2, 0.12], 
                          textColorRGB=[1, 1, 0], textSize=1.5)
        p.addUserDebugText("TARGET SHELF", [0.7, 0.3, 0.5], 
                          textColorRGB=[0, 1, 1], textSize=1.5)
        
        print("🤖 Initializing robot...")
        
        # Initialize robot
        robot = RobotController(physics_client)
        robot.initialize()
        
        # Add friction to gripper fingers for better grasping
        p.changeDynamics(robot.robot_id, 9,  # panda_finger_joint1
                        lateralFriction=1.0, 
                        spinningFriction=0.005,
                        rollingFriction=0.005)
        p.changeDynamics(robot.robot_id, 10,  # panda_finger_joint2
                        lateralFriction=1.0, 
                        spinningFriction=0.005,
                        rollingFriction=0.005)
        
        print("📷 Setting up vision system...")
        
        # Initialize camera system  
        camera_config = CameraConfiguration(
            image_width=640,
            image_height=480, 
            fov=60.0
        )
        camera_system = CameraSystem(camera_config, physics_client)
        camera_system.initialize()
        
        # Setup overhead camera for object detection
        # Position camera closer and directly above the object
        overhead_camera = camera_system.setup_top_down_camera(
            shelf_center=(0.4, -0.2, 0.04)  # Updated to match new object height
        )
        
        # Initialize object detector
        object_detector = ObjectDetector(camera_system)
        object_detector.set_overhead_camera("top_down_0")
        
        # CRITICAL: Run physics simulation step before capturing
        for _ in range(10):
            p.stepSimulation()
            
        # Test object detection AFTER objects are loaded and physics settled
        print("🎯 Testing object detection...")
        
        # Debug: Check physics client and camera positioning
        print(f"🔧 Physics client ID: {physics_client}")
        print(f"🔧 Camera system physics client: {camera_system.physics_client}")
        test_camera = camera_system.get_camera("top_down_0")
        if test_camera:
            print(f"🔧 Camera physics client: {test_camera.physics_client}")
            print(f"🔧 Camera position: {test_camera.position}")
            print(f"🔧 Camera target: {test_camera.target}")
            
            test_data = test_camera.capture_rgbd()
            print(f"📊 Camera diagnostic - Depth range: {test_data.depth_image.min():.3f} to {test_data.depth_image.max():.3f}")
            print(f"📊 RGB image shape: {test_data.rgb_image.shape}")
            
            # Check some sample depth values
            depth_sample = test_data.depth_image[240, 320]  # Center pixel
            print(f"📊 Center pixel depth: {depth_sample:.3f}")
            
        detections = object_detector.detect_objects_on_table()
        print(object_detector.get_detection_summary(detections))
        
        # Compare detected position with actual cube position for calibration
        actual_cube_pos = [0.4, -0.2, 0.025]  # Known cube position (center of 5cm cube)
        if detections:
            detected_pos = detections[0].position
            print(f"📍 Position comparison:")
            print(f"   Actual cube: ({actual_cube_pos[0]:.3f}, {actual_cube_pos[1]:.3f}, {actual_cube_pos[2]:.3f})")
            print(f"   Detected:    ({detected_pos[0]:.3f}, {detected_pos[1]:.3f}, {detected_pos[2]:.3f})")
            print(f"   Error: X={detected_pos[0]-actual_cube_pos[0]:.3f}, Y={detected_pos[1]-actual_cube_pos[1]:.3f}, Z={detected_pos[2]-actual_cube_pos[2]:.3f}")
        else:
            print("⚠️  No objects detected for position comparison")
        
        # Vision-guided motion controller (no manual coordinates!)
        motion_controller = MotionController(
            robot, 
            object_detector=object_detector  # Pure vision guidance!
        )
        
        print("✅ Setup complete!")
        print()
        
        # Demo 1: Test gripper
        print("🔧 Testing gripper...")
        robot.control_gripper(open_gripper=True)
        for _ in range(50):
            p.stepSimulation()
            time.sleep(1./1000.)  # Match physics timestep
        print("   Gripper opened")
        
        robot.control_gripper(open_gripper=False) 
        for _ in range(50):
            p.stepSimulation()
            time.sleep(1./1000.)  # Match physics timestep
        print("   Gripper closed")
        
        # Demo 2: Simple movement
        print("\n🤖 Testing movement...")
        current_pos = robot.get_end_effector_position()
        print(f"   Current: ({current_pos[0]:.2f}, {current_pos[1]:.2f}, {current_pos[2]:.2f})")
        
        # Move to a very reachable position
        target = (0.3, -0.4, 0.3)
        print(f"   Moving to: ({target[0]:.2f}, {target[1]:.2f}, {target[2]:.2f})")
        
        success = robot.move_to_position(target)
        for _ in range(100):
            p.stepSimulation()
            time.sleep(1./1000.)  # Match physics timestep
            
        final_pos = robot.get_end_effector_position()
        print(f"   Reached: ({final_pos[0]:.2f}, {final_pos[1]:.2f}, {final_pos[2]:.2f})")
        print(f"   Success: {success}")
        
        # Demo 3: Complete pick and place - SIMPLE VERSION
        print("\n🎯 Testing pick and place (SIMPLE)...")
        
        # Very conservative, reachable placement with small Z offset for better alignment
        placement_cmd = PlacementCommand(
            object_id="simple_object",
            zone_id="simple_zone",
            position=(0.7, 0.3, 0.42),  # FAR FROM ROBOT AND PICKUP, ABOVE SHELF
            orientation=(0, 0, 0, 1),
            confidence_score=0.8
        )
        
        # Apply small Z offset to grasp waypoint for better alignment
        if hasattr(motion_controller, '_grasp_z_offset'):
            motion_controller._grasp_z_offset = 0.01  # 1cm offset
        
        print("   Starting pick-and-place sequence...")
        
        start_time = time.time()
        result = motion_controller.execute_placement(placement_cmd)
        exec_time = time.time() - start_time
        
        print()
        print("📊 EXECUTION RESULTS:")
        print(f"   Success: {result.success}")
        print(f"   Time: {exec_time:.2f}s")
        print(f"   Waypoints: {result.waypoints_completed}/{result.total_waypoints}")
        
        if result.success:
            print("   🎉 COMPLETE SUCCESS! All waypoints executed:")
            print("   ✅ 1. Move to pickup")
            print("   ✅ 2. Grasp (gripper close)")  
            print("   ✅ 3. Lift up")
            print("   ✅ 4. Move to shelf")
            print("   ✅ 5. Lower down")
            print("   ✅ 6. Release (gripper open)")
            print("   ✅ 7. Move away")
            
            print(f"\n🏆 PHASE 4 IMPLEMENTATION: COMPLETE!")
            print(f"   ✅ Robot control system working")
            print(f"   ✅ 7-waypoint sequence successful")
            print(f"   ✅ Performance: {exec_time:.1f}s (target: <30s)")
            print(f"   🚀 Ready for Phase 5: Human Validation")
        else:
            print(f"   Issue: {result.failure_reason}")
            print(f"   Progress: {result.waypoints_completed}/{result.total_waypoints} waypoints completed")
            
            if result.waypoints_completed > 0:
                waypoint_names = ["Pre-grasp approach", "Move to pickup", "Grasp", "Lift up", "Move to shelf", "Lower down", "Release", "Move away"]
                print("   Steps completed:")
                for i in range(result.waypoints_completed):
                    print(f"   ✅ {i+1}. {waypoint_names[i]}")
                print("   Steps failed:")
                for i in range(result.waypoints_completed, result.total_waypoints):
                    print(f"   ❌ {i+1}. {waypoint_names[i]}")
                    
            print(f"\n⚠️  PHASE 4: Partial progress achieved")
            print(f"   ✅ Basic robot control working")
            print(f"   ⏭️  Need to resolve waypoint {result.waypoints_completed + 1} issues")
        
        print(f"\n🎮 Demo complete! Close the PyBullet window when done.")
        
        # Keep window open with optimized timestep
        while True:
            p.stepSimulation()
            time.sleep(1./1000.)  # Match physics timestep
            
    except KeyboardInterrupt:
        print("\n👋 Demo stopped")
    finally:
        robot.cleanup()
        p.disconnect()


if __name__ == "__main__":
    main()
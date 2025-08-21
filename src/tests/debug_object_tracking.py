#!/usr/bin/env python3
"""
Debug script to track object movement during grasp and lift
Verifies if the object actually moves with the robot during waypoints 2-3
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))

import pybullet as p
import pybullet_data
import time
from control import RobotController, MotionController
from placement import PlacementCommand


def track_object_position(object_id, physics_client):
    """Get object's current position"""
    pos, _ = p.getBasePositionAndOrientation(object_id, physicsClientId=physics_client)
    return pos


def main():
    print("🔍 Object Tracking Debug - Phase 4")
    print("=" * 50)
    
    # Connect to PyBullet GUI
    physics_client = p.connect(p.GUI)
    p.setGravity(0, 0, -9.81)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    
    # Better camera view for debugging
    p.resetDebugVisualizerCamera(
        cameraDistance=1.2,
        cameraYaw=45,
        cameraPitch=-25,
        cameraTargetPosition=[0.35, -0.6, 0.25]
    )
    
    try:
        # Load ground
        p.loadURDF("plane.urdf")
        
        # Add object to pick up
        object_id = p.loadURDF(
            "cube.urdf",
            [0.35, -0.6, 0.25],  # Same position as demo
            globalScaling=0.03
        )
        
        print("🤖 Initializing robot...")
        
        # Initialize robot
        robot = RobotController(physics_client)
        robot.initialize()
        
        # Initial object position
        initial_obj_pos = track_object_position(object_id, physics_client)
        print(f"📦 Initial object position: ({initial_obj_pos[0]:.3f}, {initial_obj_pos[1]:.3f}, {initial_obj_pos[2]:.3f})")
        
        # Motion controller
        motion_controller = MotionController(
            robot, 
            pickup_location=(0.35, -0.6, 0.25)
        )
        
        print("\n🎯 Testing individual waypoints with object tracking...")
        
        # Generate waypoints for debugging
        placement_pos = (0.7, 0.3, 0.42)
        waypoints = motion_controller._generate_waypoints(placement_pos)
        
        print(f"Generated {len(waypoints)} waypoints:")
        for i, (pos, action) in enumerate(waypoints):
            print(f"  {i+1}. Move to ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}) + {action}")
        
        print("\n" + "="*60)
        
        # Execute waypoints 1-3 individually with object tracking
        for i, (waypoint_pos, action) in enumerate(waypoints[:3]):
            waypoint_num = i + 1
            print(f"\n🎯 WAYPOINT {waypoint_num}: Move to ({waypoint_pos[0]:.2f}, {waypoint_pos[1]:.2f}, {waypoint_pos[2]:.2f}) + {action}")
            
            # Before waypoint
            before_obj_pos = track_object_position(object_id, physics_client)
            before_ee_pos = robot.get_end_effector_position()
            print(f"   BEFORE - Object: ({before_obj_pos[0]:.3f}, {before_obj_pos[1]:.3f}, {before_obj_pos[2]:.3f})")
            print(f"   BEFORE - End-eff: ({before_ee_pos[0]:.3f}, {before_ee_pos[1]:.3f}, {before_ee_pos[2]:.3f})")
            
            # Execute waypoint
            print(f"   🚀 Executing waypoint...")
            success = motion_controller._execute_waypoint(waypoint_pos, action)
            
            # Let simulation settle
            for _ in range(100):
                p.stepSimulation(physicsClientId=physics_client)
                time.sleep(1./240.)
            
            # After waypoint
            after_obj_pos = track_object_position(object_id, physics_client)
            after_ee_pos = robot.get_end_effector_position()
            print(f"   AFTER  - Object: ({after_obj_pos[0]:.3f}, {after_obj_pos[1]:.3f}, {after_obj_pos[2]:.3f})")
            print(f"   AFTER  - End-eff: ({after_ee_pos[0]:.3f}, {after_ee_pos[1]:.3f}, {after_ee_pos[2]:.3f})")
            
            # Calculate movements
            obj_movement = ((after_obj_pos[0] - before_obj_pos[0])**2 + 
                           (after_obj_pos[1] - before_obj_pos[1])**2 + 
                           (after_obj_pos[2] - before_obj_pos[2])**2)**0.5
            ee_movement = ((after_ee_pos[0] - before_ee_pos[0])**2 + 
                          (after_ee_pos[1] - before_ee_pos[1])**2 + 
                          (after_ee_pos[2] - before_ee_pos[2])**2)**0.5
            
            print(f"   📏 Object movement: {obj_movement:.3f}m")
            print(f"   📏 End-eff movement: {ee_movement:.3f}m")
            print(f"   ✅ Waypoint success: {success}")
            
            # Special analysis for grasp and lift
            if action == "grasp":
                is_open, gripper_width = robot.get_gripper_state()
                print(f"   🤏 Gripper state: width={gripper_width:.3f}m, open={is_open}")
                if obj_movement < 0.001:
                    print(f"   ⚠️  ISSUE: Object didn't move during grasp!")
                    
            elif action == "move" and waypoint_num == 3:  # Lift up
                if obj_movement < 0.05:  # Should lift significantly
                    print(f"   ❌ CRITICAL ISSUE: Object not lifted with robot!")
                    print(f"      Robot moved {ee_movement:.3f}m up, object moved {obj_movement:.3f}m")
                    print(f"      This explains the visual discrepancy!")
                else:
                    print(f"   ✅ Object successfully lifted with robot")
            
            print("-" * 60)
            
            # Stop if waypoint failed
            if not success:
                print(f"   🛑 Stopping at failed waypoint {waypoint_num}")
                break
        
        print(f"\n🎮 Debug complete! Close the PyBullet window when done.")
        
        # Keep window open
        while True:
            p.stepSimulation()
            time.sleep(1./120.)
            
    except KeyboardInterrupt:
        print("\n👋 Debug stopped")
    finally:
        robot.cleanup()
        p.disconnect()


if __name__ == "__main__":
    main()
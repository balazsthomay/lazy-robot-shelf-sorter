#!/usr/bin/env python3
"""
Debug script to find the optimal grasp position
Tests different approach heights and orientations for successful grasping
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))

import pybullet as p
import pybullet_data
import time
from control import RobotController


def main():
    print("🔍 Grasp Positioning Debug")
    print("=" * 40)
    
    # Connect to PyBullet GUI
    physics_client = p.connect(p.GUI)
    p.setGravity(0, 0, -9.81)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    
    # Better camera view for grasp debugging
    p.resetDebugVisualizerCamera(
        cameraDistance=0.8,
        cameraYaw=45,
        cameraPitch=-20,
        cameraTargetPosition=[0.35, -0.6, 0.25]
    )
    
    try:
        # Load ground
        p.loadURDF("plane.urdf")
        
        # Add object to pick up - SLIGHTLY TALLER for better gripper clearance
        object_id = p.loadURDF(
            "cube.urdf",
            [0.35, -0.6, 0.025],  # Lower to ground level
            globalScaling=0.05    # Slightly larger for easier grasping
        )
        
        print("🤖 Initializing robot...")
        robot = RobotController(physics_client)
        robot.initialize()
        
        # Get object info
        obj_pos, obj_orn = p.getBasePositionAndOrientation(object_id, physicsClientId=physics_client)
        print(f"📦 Object at: ({obj_pos[0]:.3f}, {obj_pos[1]:.3f}, {obj_pos[2]:.3f})")
        
        # Test different grasp heights
        test_heights = [0.030, 0.040, 0.050, 0.060]  # Above object center
        
        for i, grasp_height in enumerate(test_heights):
            print(f"\n🎯 TEST {i+1}: Grasp height {grasp_height:.3f}m")
            
            # Position end-effector directly above object
            grasp_pos = (obj_pos[0], obj_pos[1], grasp_height)
            print(f"   Target: ({grasp_pos[0]:.3f}, {grasp_pos[1]:.3f}, {grasp_pos[2]:.3f})")
            
            # Move to grasp position
            print("   🚀 Moving to grasp position...")
            success = robot.move_to_position(grasp_pos)
            
            # Let simulation settle
            for _ in range(50):
                p.stepSimulation(physicsClientId=physics_client)
                time.sleep(1./240.)
            
            actual_pos = robot.get_end_effector_position()
            print(f"   📍 Reached: ({actual_pos[0]:.3f}, {actual_pos[1]:.3f}, {actual_pos[2]:.3f})")
            print(f"   ✅ Movement: {success}")
            
            if not success:
                print("   ❌ Skipping grasp test - couldn't reach position")
                continue
            
            # Test grasp
            print("   🤏 Testing grasp...")
            robot.control_gripper(open_gripper=True)
            
            # Wait for gripper to open
            for _ in range(30):
                p.stepSimulation(physicsClientId=physics_client)
                time.sleep(1./240.)
            
            # Close gripper
            robot.control_gripper(open_gripper=False)
            
            # Wait for grasp completion
            for _ in range(60):
                p.stepSimulation(physicsClientId=physics_client)
                time.sleep(1./240.)
            
            # Check grasp result
            is_open, gripper_width = robot.get_gripper_state()
            obj_pos_after = p.getBasePositionAndOrientation(object_id, physicsClientId=physics_client)[0]
            
            print(f"   🤏 Gripper width: {gripper_width:.4f}m")
            print(f"   📦 Object moved: {(obj_pos_after[2] - obj_pos[2]):.4f}m vertically")
            
            # Our new success criteria
            grasp_success = 0.005 < gripper_width < 0.035
            print(f"   ✅ Grasp success: {grasp_success}")
            
            if grasp_success:
                print("   🎉 SUCCESS! Testing lift...")
                
                # Test lift
                lift_pos = (grasp_pos[0], grasp_pos[1], grasp_pos[2] + 0.1)
                lift_success = robot.move_to_position(lift_pos)
                
                # Let simulation settle
                for _ in range(50):
                    p.stepSimulation(physicsClientId=physics_client)
                    time.sleep(1./240.)
                
                obj_pos_lifted = p.getBasePositionAndOrientation(object_id, physicsClientId=physics_client)[0]
                lift_distance = obj_pos_lifted[2] - obj_pos[2]
                
                print(f"   📈 Object lifted: {lift_distance:.4f}m")
                print(f"   ✅ Lift success: {lift_success and lift_distance > 0.05}")
                
                if lift_success and lift_distance > 0.05:
                    print(f"   🏆 PERFECT! Height {grasp_height:.3f}m works!")
                    break
                else:
                    # Reset for next test
                    robot.control_gripper(open_gripper=True)
                    time.sleep(0.5)
            
            print("-" * 40)
        
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
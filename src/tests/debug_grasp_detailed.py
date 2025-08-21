#!/usr/bin/env python3
"""
Detailed grasp debugging - step by step analysis
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))

import pybullet as p
import pybullet_data
import time
from control import RobotController


def main():
    print("🔍 Detailed Grasp Analysis")
    print("=" * 40)
    
    # Connect to PyBullet GUI
    physics_client = p.connect(p.GUI)
    p.setGravity(0, 0, -9.81)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    
    # Camera focused on grasp area
    p.resetDebugVisualizerCamera(
        cameraDistance=0.6,
        cameraYaw=30,
        cameraPitch=-15,
        cameraTargetPosition=[0.5, -0.3, 0.075]
    )
    
    try:
        # Load ground
        p.loadURDF("plane.urdf")
        
        # Add object - same as demo (updated)
        object_id = p.loadURDF(
            "cube.urdf",
            [0.4, -0.2, 0.05],
            globalScaling=0.05
        )
        
        print("🤖 Initializing robot...")
        robot = RobotController(physics_client)
        robot.initialize()
        
        # Target position - same as demo (updated)
        target_pos = (0.4, -0.2, 0.05)
        
        print(f"📍 Target grasp position: {target_pos}")
        print(f"📦 Object position: {p.getBasePositionAndOrientation(object_id)[0]}")
        
        # Step 1: Move to position
        print("\n🚀 STEP 1: Moving to grasp position...")
        success = robot.move_to_position(target_pos)
        
        # Let settle
        for _ in range(100):
            p.stepSimulation()
            time.sleep(1./240.)
        
        actual_pos = robot.get_end_effector_position()
        print(f"   Target: {target_pos}")
        print(f"   Actual: ({actual_pos[0]:.3f}, {actual_pos[1]:.3f}, {actual_pos[2]:.3f})")
        
        pos_error = ((actual_pos[0] - target_pos[0])**2 + 
                    (actual_pos[1] - target_pos[1])**2 + 
                    (actual_pos[2] - target_pos[2])**2)**0.5
        print(f"   Error: {pos_error:.3f}m")
        print(f"   Success: {success}")
        
        if not success:
            print("❌ Cannot proceed - position unreachable")
            return
        
        # Step 2: Open gripper
        print("\n🤏 STEP 2: Opening gripper...")
        robot.control_gripper(open_gripper=True)
        
        for _ in range(50):
            p.stepSimulation()
            time.sleep(1./240.)
        
        is_open, width = robot.get_gripper_state()
        print(f"   Gripper open: {is_open}, width: {width:.4f}m")
        
        # Step 3: Close gripper (attempt grasp)
        print("\n🤏 STEP 3: Closing gripper (grasp attempt)...")
        obj_pos_before = p.getBasePositionAndOrientation(object_id)[0]
        
        robot.control_gripper(open_gripper=False)
        
        # Monitor grasp process
        for i in range(60):
            p.stepSimulation()
            time.sleep(1./240.)
            
            if i % 20 == 0:  # Check every 20 steps
                is_open, width = robot.get_gripper_state()
                obj_pos_current = p.getBasePositionAndOrientation(object_id)[0]
                print(f"   Step {i}: width={width:.4f}m, obj_z={obj_pos_current[2]:.4f}m")
        
        # Final grasp analysis
        is_open, final_width = robot.get_gripper_state()
        obj_pos_after = p.getBasePositionAndOrientation(object_id)[0]
        
        print(f"\n📊 GRASP ANALYSIS:")
        print(f"   Final gripper width: {final_width:.4f}m")
        print(f"   Object moved: {(obj_pos_after[2] - obj_pos_before[2]):.4f}m vertically")
        print(f"   Our success criteria: 0.005 < width < 0.035")
        print(f"   Meets criteria: {0.005 < final_width < 0.035}")
        
        # Check contact points
        contacts = p.getContactPoints(robot.robot_id, object_id, physicsClientId=physics_client)
        print(f"   Contact points: {len(contacts)}")
        
        if len(contacts) > 0:
            print("   🎯 Robot is touching object!")
            for i, contact in enumerate(contacts[:3]):  # Show first 3 contacts
                print(f"      Contact {i+1}: link {contact[3]} at {contact[5]}")
        else:
            print("   ⚠️ No contact between robot and object")
        
        # Step 4: Test lift (if grasp seemed successful)
        if 0.005 < final_width < 0.035:
            print(f"\n🚀 STEP 4: Testing lift...")
            lift_pos = (target_pos[0], target_pos[1], target_pos[2] + 0.1)
            lift_success = robot.move_to_position(lift_pos)
            
            for _ in range(100):
                p.stepSimulation()
                time.sleep(1./240.)
            
            obj_pos_lifted = p.getBasePositionAndOrientation(object_id)[0]
            lift_distance = obj_pos_lifted[2] - obj_pos_before[2]
            
            print(f"   Object lifted: {lift_distance:.4f}m")
            print(f"   Lift success: {lift_success}")
            
            if lift_distance > 0.05:
                print("   🎉 COMPLETE SUCCESS!")
            else:
                print("   ❌ Object not lifted despite grasp")
        else:
            print(f"\n❌ Grasp failed - skipping lift test")
        
        print(f"\n🎮 Analysis complete! Close window when done.")
        
        # Keep window open
        while True:
            p.stepSimulation()
            time.sleep(1./120.)
            
    except KeyboardInterrupt:
        print("\n👋 Analysis stopped")
    finally:
        robot.cleanup()
        p.disconnect()


if __name__ == "__main__":
    main()
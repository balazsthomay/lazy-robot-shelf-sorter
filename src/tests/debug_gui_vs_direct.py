#!/usr/bin/env python3
"""
Minimal test to compare DIRECT vs GUI mode robot behavior
Tests the hypothesis that robot works in DIRECT but not GUI mode
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pybullet as p
import pybullet_data
import time
import numpy as np
from control import RobotController

def test_robot_mode(mode_name, connect_mode):
    """Test robot in specified PyBullet mode"""
    print(f"\n🔧 Testing Robot in {mode_name} mode")
    print("=" * 40)
    
    # Connect with specified mode
    physics_client = p.connect(connect_mode)
    p.setGravity(0, 0, -9.81)
    
    # CRITICAL TEST: Add missing setTimeStep for GUI mode
    if connect_mode == p.GUI:
        p.setTimeStep(1./240.)  # Missing from demo!
        print("   ✅ Added setTimeStep for GUI mode")
    
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF("plane.urdf")
    
    try:
        # Initialize robot
        robot = RobotController(physics_client)
        robot.initialize()
        
        print(f"   🤖 Robot initialized with {robot.num_joints} joints")
        
        # Record initial joint positions
        initial_joints = []
        for i in range(7):  # Arm joints 0-6
            joint_state = p.getJointState(robot.robot_id, i, physicsClientId=physics_client)
            initial_joints.append(joint_state[0])
        
        print(f"   📍 Initial joint positions: {[f'{j:.3f}' for j in initial_joints[:3]]}")
        
        # Test gripper movement
        print(f"   🔧 Testing gripper control...")
        robot.control_gripper(open_gripper=True)
        
        # Step simulation adequately
        for _ in range(120):  # Give enough time
            p.stepSimulation(physicsClientId=physics_client)
            if connect_mode == p.GUI:
                time.sleep(1./240.)  # Only sleep in GUI mode
        
        # Check if joints actually moved
        final_joints = []
        max_joint_movement = 0.0
        
        for i in range(7):
            joint_state = p.getJointState(robot.robot_id, i, physicsClientId=physics_client)
            final_joints.append(joint_state[0])
            movement = abs(final_joints[i] - initial_joints[i])
            max_joint_movement = max(max_joint_movement, movement)
        
        print(f"   📍 Final joint positions:   {[f'{j:.3f}' for j in final_joints[:3]]}")
        print(f"   📏 Max joint movement: {max_joint_movement:.6f} radians")
        
        # Test arm movement
        print(f"   🤖 Testing arm movement...")
        target_pos = (0.3, -0.3, 0.4)
        
        success = robot.move_to_position(target_pos, timeout=5.0)
        
        # Additional simulation steps for GUI mode
        if connect_mode == p.GUI:
            print(f"   ⏰ Additional GUI simulation steps...")
            for _ in range(240):  # 1 second more at 240Hz
                p.stepSimulation(physicsClientId=physics_client)
                time.sleep(1./240.)
        
        final_pos = robot.get_end_effector_position()
        
        print(f"   📊 RESULTS:")
        print(f"   Success reported: {success}")
        print(f"   Max joint movement: {max_joint_movement:.6f} rad")
        print(f"   End effector moved to: {final_pos}")
        
        # Determine if robot actually moved
        actually_moved = max_joint_movement > 0.01  # More than 0.6 degrees
        
        if actually_moved:
            print(f"   🎉 ROBOT ACTUALLY MOVED in {mode_name} mode")
        else:
            print(f"   ❌ ROBOT DID NOT MOVE in {mode_name} mode")
        
        robot.cleanup()
        return actually_moved
        
    except Exception as e:
        print(f"   💥 ERROR in {mode_name} mode: {str(e)}")
        return False
    finally:
        p.disconnect()

def main():
    """Compare robot behavior in DIRECT vs GUI modes"""
    print("🚀 ROBOT MODE COMPARISON TEST")
    print("Testing hypothesis: Robot works in DIRECT but not GUI mode")
    
    # Test 1: DIRECT mode (how built-in test works)
    direct_works = test_robot_mode("DIRECT", p.DIRECT)
    
    # Test 2: GUI mode with proper physics setup
    gui_works = test_robot_mode("GUI", p.GUI)
    
    print("\n📊 FINAL COMPARISON:")
    print(f"   DIRECT mode works: {'✅ YES' if direct_works else '❌ NO'}")
    print(f"   GUI mode works:    {'✅ YES' if gui_works else '❌ NO'}")
    
    if direct_works and not gui_works:
        print("\n🎯 HYPOTHESIS CONFIRMED: Robot works in DIRECT but not GUI")
        print("   Root cause: GUI mode needs different physics setup")
    elif direct_works and gui_works:
        print("\n🎉 HYPOTHESIS DISPROVEN: Both modes work with proper setup")
        print("   The missing setTimeStep was the issue!")
    else:
        print("\n🤔 UNEXPECTED RESULT: Further investigation needed")

if __name__ == "__main__":
    main()
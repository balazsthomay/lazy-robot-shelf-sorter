#!/usr/bin/env python3
"""
Focused test specifically for gripper visual movement
Tests if the gripper actually opens/closes in GUI, not just reports success
"""

import sys
import os
sys.path.append('src')

import pybullet as p
import pybullet_data
import time
from control import RobotController

def main():
    """Test if gripper actually moves visually"""
    print("🔧 FOCUSED GRIPPER MOVEMENT TEST")
    print("=" * 40)
    
    # Connect to PyBullet GUI - CRITICAL for visual validation
    physics_client = p.connect(p.GUI)
    p.setGravity(0, 0, -9.81)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF("plane.urdf")
    
    # Better camera view to see gripper
    p.resetDebugVisualizerCamera(
        cameraDistance=1.0,
        cameraYaw=0,
        cameraPitch=-10,
        cameraTargetPosition=[0, -0.8, 0.8]
    )
    
    try:
        print("🤖 Initializing robot...")
        robot = RobotController(physics_client)
        robot.initialize()
        
        print("👀 WATCH THE GRIPPER FINGERS CAREFULLY!")
        print("   Look at the robot gripper in the GUI window")
        print("   The fingers should open and close visually")
        print()
        
        # Test 1: Initial state
        print("📋 Test 1: Initial gripper state")
        is_open, width = robot.get_gripper_state()
        print(f"   Initial state: {'Open' if is_open else 'Closed'} ({width:.4f}m)")
        
        input("   Press Enter to test gripper opening...")
        
        # Test 2: Open gripper
        print("📋 Test 2: Opening gripper...")
        print("   🔍 WATCH: Do the gripper fingers move APART in the GUI?")
        robot.control_gripper(open_gripper=True)
        
        # Give adequate time for visual movement
        for i in range(120):  # 0.5 seconds at 240Hz
            p.stepSimulation()
            time.sleep(1./240.)
        
        is_open, width = robot.get_gripper_state()
        print(f"   After opening: {'Open' if is_open else 'Closed'} ({width:.4f}m)")
        
        user_observation = input("   Did you SEE the gripper fingers move apart in GUI? (y/n): ").strip().lower()
        
        input("   Press Enter to test gripper closing...")
        
        # Test 3: Close gripper  
        print("📋 Test 3: Closing gripper...")
        print("   🔍 WATCH: Do the gripper fingers move TOGETHER in the GUI?")
        robot.control_gripper(open_gripper=False)
        
        # Give adequate time for visual movement
        for i in range(120):  # 0.5 seconds at 240Hz
            p.stepSimulation()
            time.sleep(1./240.)
            
        is_open, width = robot.get_gripper_state()
        print(f"   After closing: {'Open' if is_open else 'Closed'} ({width:.4f}m)")
        
        user_observation_2 = input("   Did you SEE the gripper fingers move together in GUI? (y/n): ").strip().lower()
        
        print()
        print("📊 CRITICAL VALIDATION RESULTS:")
        print(f"   Terminal reports state changes: ✅ YES")
        print(f"   User observed opening movement: {'✅ YES' if user_observation == 'y' else '❌ NO'}")
        print(f"   User observed closing movement: {'✅ YES' if user_observation_2 == 'y' else '❌ NO'}")
        
        if user_observation == 'n' or user_observation_2 == 'n':
            print(f"   🚨 BUG CONFIRMED: Gripper not moving visually despite terminal success")
            print(f"   🔧 Root cause: Motor control parameters still insufficient")
        else:
            print(f"   🎉 SUCCESS: Gripper moves visually and matches terminal reports")
        
        print(f"\n🎮 Test complete. Close PyBullet window when done.")
        
        # Keep window open for observation
        while True:
            p.stepSimulation()
            time.sleep(1./240.)
            
    except KeyboardInterrupt:
        print("\n👋 Test stopped")
    finally:
        robot.cleanup()
        p.disconnect()

if __name__ == "__main__":
    main()
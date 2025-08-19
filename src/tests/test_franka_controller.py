#!/usr/bin/env python3
"""
Test RobotController with Franka Panda robot
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pybullet as p
import pybullet_data
from control import RobotController

def test_franka_controller():
    """Test that RobotController works correctly with Franka Panda"""
    print("🤖 Testing RobotController with Franka Panda...")
    
    # Connect headless
    physics_client = p.connect(p.DIRECT)
    p.setGravity(0, 0, -9.81)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    
    # Load ground
    ground_id = p.loadURDF("plane.urdf")
    
    try:
        # Test RobotController initialization
        print("\n📦 Creating RobotController...")
        robot = RobotController(physics_client=physics_client)
        robot.initialize()
        
        print(f"✅ Robot loaded successfully!")
        print(f"   Total joints: {robot.num_joints}")
        print(f"   Controllable joints: {len(robot.joint_indices)}")
        print(f"   Joint indices: {robot.joint_indices}")
        print(f"   Arm end-effector link: {robot.arm_end_effector_link}")
        
        # Test initial position
        print("\n📍 Testing end-effector position...")
        initial_pos = robot.get_end_effector_position()
        print(f"   Initial position: {initial_pos}")
        
        # Test movement
        print("\n🎯 Testing robot movement...")
        targets = [
            (0.5, 0.0, 0.5),   # Forward
            (0.0, 0.5, 0.5),   # Side
            (0.3, -0.3, 0.8),  # Up and forward
            (0.0, -0.8, 0.5),  # Back to near start
        ]
        
        for i, target in enumerate(targets):
            print(f"   Target {i+1}: {target}")
            
            success = robot.move_to_position(target)
            if success:
                # Run simulation to let robot move
                for _ in range(50):
                    p.stepSimulation()
                
                final_pos = robot.get_end_effector_position()
                distance = ((target[0] - final_pos[0])**2 + 
                          (target[1] - final_pos[1])**2 + 
                          (target[2] - final_pos[2])**2)**0.5
                
                print(f"     ✅ Movement succeeded")
                print(f"     Final position: {final_pos}")
                print(f"     Distance error: {distance:.3f}m")
                
                if distance < 0.1:  # Within 10cm
                    print(f"     ✅ Accurate positioning!")
                else:
                    print(f"     ⚠️ Large positioning error")
            else:
                print(f"     ❌ Movement failed")
        
        # Test joint control
        print("\n🔧 Testing individual joint control...")
        if robot.joint_indices:
            # Move first joint
            joint_idx = robot.joint_indices[0]
            print(f"   Moving joint {joint_idx} to 0.5 radians...")
            
            p.setJointMotorControl2(
                robot.robot_id,
                joint_idx,
                p.POSITION_CONTROL,
                targetPosition=0.5,
                physicsClientId=physics_client
            )
            
            # Let it move
            for _ in range(100):
                p.stepSimulation()
            
            joint_state = p.getJointState(robot.robot_id, joint_idx, physicsClientId=physics_client)
            print(f"   Final joint position: {joint_state[0]:.3f}")
            print("   ✅ Joint control working!")
        
        # Test cleanup
        print("\n🧹 Testing cleanup...")
        robot.cleanup()
        print("   ✅ Cleanup completed")
        
        p.disconnect()
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        p.disconnect()
        return False

if __name__ == "__main__":
    success = test_franka_controller()
    
    if success:
        print("\n🎉 Franka Panda RobotController test passed!")
        print("✅ Ready to use in demos and Phase 2")
    else:
        print("\n❌ Franka Panda RobotController test failed!")
        print("Need to fix issues before proceeding")
#!/usr/bin/env python3
"""
Debug motor control parameters by testing different force/velocity values
Tests hypothesis that motor parameters are insufficient for actual movement
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pybullet as p
import pybullet_data
import time
import numpy as np

def test_motor_params(force, max_velocity, position_gain, velocity_gain):
    """Test specific motor control parameters"""
    print(f"\n🔧 Testing: force={force}, maxVel={max_velocity}, posGain={position_gain}, velGain={velocity_gain}")
    
    # Connect to GUI for visual confirmation
    physics_client = p.connect(p.GUI)
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(1./240.)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF("plane.urdf")
    
    # Camera positioned to see robot clearly
    p.resetDebugVisualizerCamera(
        cameraDistance=1.5,
        cameraYaw=45,
        cameraPitch=-20,
        cameraTargetPosition=[0, -0.8, 0.5]
    )
    
    try:
        # Load robot
        robot_id = p.loadURDF("franka_panda/panda.urdf", [0, -0.8, 0], useFixedBase=True)
        
        # Record initial joint positions
        initial_joints = []
        for i in range(7):
            joint_state = p.getJointState(robot_id, i)
            initial_joints.append(joint_state[0])
        
        print(f"   📍 Initial joints: {[f'{j:.3f}' for j in initial_joints[:3]]}")
        
        # Apply strong motor control to joint 1 (should be very visible)
        target_angle = 0.5  # 28.6 degrees - should be clearly visible
        
        p.setJointMotorControl2(
            robot_id,
            1,  # Joint 1 (shoulder joint - very visible)
            p.POSITION_CONTROL,
            targetPosition=target_angle,
            force=force,
            maxVelocity=max_velocity,
            positionGain=position_gain,
            velocityGain=velocity_gain
        )
        
        print(f"   🎯 Target joint 1 angle: {target_angle:.3f} rad ({target_angle*57.3:.1f}°)")
        print(f"   👀 WATCH joint 1 (shoulder) in GUI - should rotate!")
        
        # Step simulation with adequate time
        steps = 480  # 2 seconds at 240Hz
        for i in range(steps):
            p.stepSimulation()
            time.sleep(1./240.)
            
            # Check progress every 120 steps (0.5 seconds)
            if i % 120 == 0:
                joint_state = p.getJointState(robot_id, 1)
                current_angle = joint_state[0]
                progress = abs(current_angle - initial_joints[1])
                print(f"   📊 Step {i}: joint angle={current_angle:.3f}, progress={progress:.3f}")
        
        # Final measurement
        final_joints = []
        for i in range(7):
            joint_state = p.getJointState(robot_id, i)
            final_joints.append(joint_state[0])
        
        movement = abs(final_joints[1] - initial_joints[1])
        error = abs(final_joints[1] - target_angle)
        
        print(f"   📍 Final joint 1: {final_joints[1]:.3f} rad ({final_joints[1]*57.3:.1f}°)")
        print(f"   📏 Movement achieved: {movement:.6f} rad ({movement*57.3:.3f}°)")
        print(f"   🎯 Error from target: {error:.6f} rad ({error*57.3:.3f}°)")
        
        success = movement > 0.05  # More than 2.9 degrees of movement
        print(f"   📊 RESULT: {'✅ SUCCESS' if success else '❌ FAILED'} - Joint moved significantly")
        
        # Keep window open for visual inspection
        input("   Press Enter to close this test...")
        
        return success, movement
        
    except Exception as e:
        print(f"   💥 ERROR: {str(e)}")
        return False, 0.0
    finally:
        p.disconnect()

def main():
    """Test different motor control parameter combinations"""
    print("🚀 MOTOR CONTROL PARAMETER DEBUG TEST")
    print("Testing hypothesis: Motor parameters insufficient for movement")
    
    # Test different parameter combinations
    test_configs = [
        # (force, maxVelocity, positionGain, velocityGain)
        (50.0, 0.5, 0.1, 0.5),    # Weak parameters
        (200.0, 1.0, 0.3, 1.0),   # Current parameters  
        (500.0, 2.0, 0.5, 1.0),   # Strong parameters
        (1000.0, 3.0, 1.0, 1.0),  # Very strong parameters
        (2000.0, 5.0, 2.0, 2.0),  # Maximum parameters
    ]
    
    results = []
    
    for i, (force, vel, pos_gain, vel_gain) in enumerate(test_configs):
        print(f"\n{'='*60}")
        print(f"TEST {i+1}/{len(test_configs)}")
        
        success, movement = test_motor_params(force, vel, pos_gain, vel_gain)
        results.append((success, movement, force, vel))
        
        if success:
            print(f"🎉 FOUND WORKING PARAMETERS!")
            print(f"   force={force}, maxVel={vel}, movement={movement:.3f} rad")
            break
    
    print(f"\n📊 FINAL RESULTS:")
    for i, (success, movement, force, vel) in enumerate(results):
        status = "✅ WORKS" if success else "❌ FAILS"
        print(f"   Config {i+1}: {status} - force={force}, movement={movement:.3f}rad")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Robot Validation - Basic robot integration test
Part of Phase 1: Foundation - Milestone 1

"""

import time
import pybullet as p
import pybullet_data


def validate_robot_integration() -> bool:
    """Single function to validate robot works"""
    print("🚀 Robot integration validation...")
    start_time = time.time()
    
    # Initialize
    physics_client = p.connect(p.DIRECT)  # KISS: no GUI
    p.setGravity(0, 0, -9.81)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    
    # Load ground
    p.loadURDF("plane.urdf")
    
    # Load robot - YAGNI: use built-in kuka robot (similar to Franka)
    try:
        robot_id = p.loadURDF("kuka_iiwa/model.urdf", [0, 0, 0])
        print("✅ Robot loaded successfully")
    except Exception as e:
        print(f"❌ Robot loading failed: {e}")
        p.disconnect()
        return False
    
    # Basic movement test - KISS: just move one joint
    num_joints = p.getNumJoints(robot_id)
    if num_joints > 0:
        # Move first joint
        p.setJointMotorControl2(robot_id, 0, p.POSITION_CONTROL, targetPosition=0.5)
        
        # Run simulation briefly
        for _ in range(100):
            p.stepSimulation()
            
        # Check if joint moved
        joint_state = p.getJointState(robot_id, 0)
        joint_position = joint_state[0]
        
        if abs(joint_position - 0.5) < 0.1:
            print("✅ Robot movement validated")
            success = True
        else:
            print(f"❌ Robot movement failed - position: {joint_position}")
            success = False
    else:
        print("❌ No joints found")
        success = False
    
    total_time = time.time() - start_time
    print(f"✅ Robot validation time: {total_time:.3f}s")
    
    # Cleanup
    p.disconnect()
    
    return success and total_time < 3.0


def main():
    """Main function"""
    success = validate_robot_integration()
    
    if success:
        print("🎉 Robot integration SUCCESSFUL")
    else:
        print("❌ Robot integration FAILED")
        
    return success


if __name__ == "__main__":
    main()
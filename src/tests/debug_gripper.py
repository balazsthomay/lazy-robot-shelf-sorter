#!/usr/bin/env python3
"""
Debug gripper joints to understand Franka Panda structure
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pybullet as p
import pybullet_data
from control import RobotController


def debug_robot_joints():
    """Debug robot joint structure"""
    print("🔍 Debugging Franka Panda joint structure...")
    
    # Initialize PyBullet
    physics_client = p.connect(p.DIRECT)
    p.setGravity(0, 0, -9.81)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    
    try:
        # Load robot
        robot = RobotController(physics_client)
        robot.initialize()
        
        print(f"Total joints: {robot.num_joints}")
        print(f"Controllable joints: {len(robot.joint_indices)}")
        print()
        
        # Print all joint info
        for i in range(robot.num_joints):
            joint_info = p.getJointInfo(robot.robot_id, i, physicsClientId=physics_client)
            joint_name = joint_info[1].decode('utf-8')
            joint_type = joint_info[2]
            joint_lower = joint_info[8]
            joint_upper = joint_info[9]
            
            type_name = {
                p.JOINT_REVOLUTE: "REVOLUTE",
                p.JOINT_PRISMATIC: "PRISMATIC", 
                p.JOINT_FIXED: "FIXED"
            }.get(joint_type, f"TYPE_{joint_type}")
            
            print(f"Joint {i}: {joint_name} [{type_name}] limits: [{joint_lower:.3f}, {joint_upper:.3f}]")
        
        robot.cleanup()
        
    finally:
        p.disconnect()


if __name__ == "__main__":
    debug_robot_joints()
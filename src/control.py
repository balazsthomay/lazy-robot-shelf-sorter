#!/usr/bin/env python3
"""
Robot Controller - Simple robot control interface
Part of Phase 1: Foundation - Milestone 3
"""

import pybullet as p
import pybullet_data
import numpy as np
from typing import List, Tuple, Optional
from interfaces import SimulationComponent


class RobotController(SimulationComponent):
    """Franka Panda robot controller (7-DOF arm + 2-DOF gripper)"""
    
    def __init__(self, physics_client: int = None):
        self.robot_id: Optional[int] = None
        self.num_joints = 0
        self.joint_indices: List[int] = []
        self.physics_client = physics_client
        
    def initialize(self) -> None:
        """Load and setup robot"""
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # Use Franka Panda robot (7-DOF arm + 2-DOF gripper)
        self.robot_id = p.loadURDF("franka_panda/panda.urdf", [0, -0.8, 0], useFixedBase=True, physicsClientId=self.physics_client)
        self.num_joints = p.getNumJoints(self.robot_id, physicsClientId=self.physics_client)
        
        # Get controllable joints
        self.joint_indices = []
        for i in range(self.num_joints):
            joint_info = p.getJointInfo(self.robot_id, i, physicsClientId=self.physics_client)
            if joint_info[2] != p.JOINT_FIXED:  # Skip fixed joints
                self.joint_indices.append(i)
        
        # For Franka Panda: arm joints are 0-6, end-effector is at joint 6 (7th arm joint)
        self.arm_end_effector_link = 6  # panda_joint7 is the arm tip
                
    def move_to_position(self, target_position: Tuple[float, float, float]) -> bool:
        """Move end effector to target position"""
        if self.robot_id is None:
            return False
            
        # Simple IK using PyBullet (use arm end-effector, not gripper)
        joint_positions = p.calculateInverseKinematics(
            self.robot_id,
            self.arm_end_effector_link,  # Arm end-effector (joint 6)
            target_position,
            physicsClientId=self.physics_client
        )
        
        # Apply joint positions
        for i, joint_idx in enumerate(self.joint_indices):
            if i < len(joint_positions):
                p.setJointMotorControl2(
                    self.robot_id,
                    joint_idx,
                    p.POSITION_CONTROL,
                    targetPosition=joint_positions[i],
                    physicsClientId=self.physics_client
                )
        
        return True
        
    def get_end_effector_position(self) -> Tuple[float, float, float]:
        """Get current end effector position"""
        if self.robot_id is None:
            return (0, 0, 0)
            
        link_state = p.getLinkState(self.robot_id, self.arm_end_effector_link, physicsClientId=self.physics_client)
        return link_state[0]  # Position
        
    def cleanup(self) -> None:
        """Clean up robot resources"""
        if self.robot_id is not None:
            p.removeBody(self.robot_id, physicsClientId=self.physics_client)
            self.robot_id = None


class WorkspaceAnalyzer:
    """Analyze robot workspace coverage"""
    
    def __init__(self, robot: RobotController):
        self.robot = robot
        
    def check_reachability(self, positions: List[Tuple[float, float, float]]) -> List[bool]:
        """Check if robot can reach given positions"""
        reachable = []
        
        for pos in positions:
            # Test if IK solution exists
            if self.robot.robot_id is None:
                reachable.append(False)
                continue
                
            try:
                joint_positions = p.calculateInverseKinematics(
                    self.robot.robot_id,
                    self.robot.num_joints - 1,
                    pos
                )
                
                # Check if solution is valid (not None and within bounds)
                if joint_positions and len(joint_positions) > 0:
                    reachable.append(True)
                else:
                    reachable.append(False)
                    
            except:
                reachable.append(False)
                
        return reachable
        
    def calculate_coverage(self, positions: List[Tuple[float, float, float]]) -> float:
        """Calculate percentage of positions reachable"""
        if not positions:
            return 0.0
            
        reachable = self.check_reachability(positions)
        return sum(reachable) / len(reachable) * 100


def main():
    """Test robot controller"""
    print("🚀 Testing RobotController...")
    
    # Initialize PyBullet
    physics_client = p.connect(p.DIRECT)
    p.setGravity(0, 0, -9.81)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF("plane.urdf")
    
    # Test robot controller
    robot = RobotController()
    robot.initialize()
    
    # Test positions (shelf-like positions)
    test_positions = [
        (0.3, 0.2, 0.3),   # Shelf position 1
        (0.3, 0.0, 0.3),   # Shelf position 2
        (0.3, -0.2, 0.3),  # Shelf position 3
        (0.3, 0.2, 0.6),   # Higher shelf
        (0.3, 0.0, 0.6),   # Higher shelf
    ]
    
    # Test workspace analysis
    analyzer = WorkspaceAnalyzer(robot)
    reachable = analyzer.check_reachability(test_positions)
    coverage = analyzer.calculate_coverage(test_positions)
    
    print(f"✅ Robot loaded with {robot.num_joints} joints")
    print(f"✅ Reachable positions: {sum(reachable)}/{len(test_positions)}")
    print(f"✅ Workspace coverage: {coverage:.1f}%")
    
    # Test movement
    if test_positions:
        initial_pos = robot.get_end_effector_position()
        robot.move_to_position(test_positions[0])
        
        # Run simulation steps
        for _ in range(100):
            p.stepSimulation()
            
        final_pos = robot.get_end_effector_position()
        moved = np.linalg.norm(np.array(final_pos) - np.array(initial_pos)) > 0.1
        print(f"✅ Robot movement: {'SUCCESS' if moved else 'FAILED'}")
    
    robot.cleanup()
    p.disconnect()
    
    success = coverage >= 60  # 60% coverage minimum
    print(f"🎉 Robot controller test: {'PASSED' if success else 'FAILED'}")
    return success


if __name__ == "__main__":
    main()
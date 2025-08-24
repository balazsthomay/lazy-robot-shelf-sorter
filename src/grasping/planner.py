"""
Motion Planning for Robot Grasping System

Extends existing robot controller with ML-guided motion planning,
collision checking, and orientation constraints from GG-CNN predictions.
"""

import numpy as np
import pybullet as p
from typing import List, Optional, Tuple
from dataclasses import dataclass
import logging

from .predictor import GraspPose

logger = logging.getLogger(__name__)


@dataclass
class JointTrajectory:
    """Represents a planned joint trajectory for grasp execution."""
    joint_positions: List[np.ndarray]  # List of 7-DOF joint positions
    gripper_widths: List[float]        # Gripper opening at each waypoint
    timestamps: List[float]            # Time for each waypoint
    success: bool                      # Whether planning succeeded
    
    
class MotionPlanner:
    """
    Motion planning with collision checking for grasping operations.
    
    Extends existing RobotController with ML-guided motions and
    orientation constraints from GG-CNN predictions.
    """
    
    def __init__(self, robot_id: int, end_effector_link: int = 7):
        """
        Initialize motion planner.
        
        Args:
            robot_id: PyBullet robot body ID
            end_effector_link: Link index for end effector
        """
        self.robot_id = robot_id
        self.end_effector_link = end_effector_link
        
        # Joint limits for Franka Panda
        self.joint_limits_lower = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
        self.joint_limits_upper = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
        
        # Planning parameters
        self.max_ik_iterations = 100
        self.position_tolerance = 0.01  # 1cm
        self.orientation_tolerance = 0.1  # ~5.7 degrees
        
        logger.info(f"Initialized motion planner for robot {robot_id}")
        
    def validate_reachability(self, grasp_pose: GraspPose) -> bool:
        """
        Check if grasp pose is reachable by the robot.
        
        Args:
            grasp_pose: Target grasp pose
            
        Returns:
            True if pose is reachable
        """
        try:
            # Quick IK check without trajectory planning
            joint_positions = p.calculateInverseKinematics(
                self.robot_id,
                self.end_effector_link,
                grasp_pose.position,
                grasp_pose.orientation,
                lowerLimits=self.joint_limits_lower.tolist(),
                upperLimits=self.joint_limits_upper.tolist(),
                jointRanges=(self.joint_limits_upper - self.joint_limits_lower).tolist(),
                maxNumIterations=self.max_ik_iterations
            )
            
            # Check if solution is within joint limits
            joint_positions = np.array(joint_positions[:7])  # Only arm joints
            within_limits = np.all(
                (joint_positions >= self.joint_limits_lower) & 
                (joint_positions <= self.joint_limits_upper)
            )
            
            if not within_limits:
                logger.debug(f"Grasp pose unreachable: joint limits violated")
                return False
                
            # Verify forward kinematics accuracy
            fk_result = p.getLinkState(self.robot_id, self.end_effector_link)
            achieved_pos = np.array(fk_result[0])
            achieved_orn = np.array(fk_result[1])
            
            pos_error = np.linalg.norm(achieved_pos - grasp_pose.position)
            
            # Simple orientation error check (could be improved)
            orn_error = np.linalg.norm(achieved_orn - grasp_pose.orientation)
            
            reachable = (pos_error < self.position_tolerance and 
                        orn_error < self.orientation_tolerance)
            
            if not reachable:
                logger.debug(f"Grasp pose unreachable: IK error too high (pos: {pos_error:.3f}, orn: {orn_error:.3f})")
                
            return reachable
            
        except Exception as e:
            logger.debug(f"Reachability check failed: {e}")
            return False
            
    def check_collision(self, joint_positions: np.ndarray) -> bool:
        """
        Check for collisions at given joint configuration.
        
        Args:
            joint_positions: 7-DOF joint positions
            
        Returns:
            True if collision detected
        """
        try:
            # Set robot to test configuration
            for i, pos in enumerate(joint_positions):
                p.resetJointState(self.robot_id, i, pos)
                
            # Check for collisions
            contacts = p.getContactPoints(bodyA=self.robot_id)
            
            # Filter out self-collisions and ground contact
            for contact in contacts:
                bodyB = contact[2]
                if bodyB != self.robot_id and bodyB != 0:  # Not self or ground
                    logger.debug(f"Collision detected with body {bodyB}")
                    return True
                    
            return False
            
        except Exception as e:
            logger.debug(f"Collision check failed: {e}")
            return True  # Conservative: assume collision if check fails
            
    def plan_grasp_approach(self, grasp_pose: GraspPose) -> JointTrajectory:
        """
        Plan approach trajectory to grasp pose.
        
        Args:
            grasp_pose: Target grasp pose
            
        Returns:
            Planned joint trajectory
        """
        if not self.validate_reachability(grasp_pose):
            return JointTrajectory([], [], [], success=False)
            
        try:
            # Get current joint positions
            current_joints = []
            for i in range(7):
                joint_state = p.getJointState(self.robot_id, i)
                current_joints.append(joint_state[0])
            current_joints = np.array(current_joints)
            
            # Plan pre-grasp pose (10cm above target)
            pre_grasp_position = grasp_pose.position + np.array([0, 0, 0.10])
            pre_grasp_pose = GraspPose(
                position=pre_grasp_position,
                orientation=grasp_pose.orientation,
                width=grasp_pose.width,
                confidence=grasp_pose.confidence
            )
            
            # Solve IK for pre-grasp
            pre_grasp_joints = p.calculateInverseKinematics(
                self.robot_id,
                self.end_effector_link,
                pre_grasp_pose.position,
                pre_grasp_pose.orientation,
                lowerLimits=self.joint_limits_lower.tolist(),
                upperLimits=self.joint_limits_upper.tolist(),
                jointRanges=(self.joint_limits_upper - self.joint_limits_lower).tolist(),
                maxNumIterations=self.max_ik_iterations
            )
            pre_grasp_joints = np.array(pre_grasp_joints[:7])
            
            # Solve IK for final grasp
            final_joints = p.calculateInverseKinematics(
                self.robot_id,
                self.end_effector_link,
                grasp_pose.position,
                grasp_pose.orientation,
                lowerLimits=self.joint_limits_lower.tolist(),
                upperLimits=self.joint_limits_upper.tolist(),
                jointRanges=(self.joint_limits_upper - self.joint_limits_lower).tolist(),
                maxNumIterations=self.max_ik_iterations
            )
            final_joints = np.array(final_joints[:7])
            
            # Check for collisions at key waypoints
            if self.check_collision(pre_grasp_joints) or self.check_collision(final_joints):
                logger.debug("Collision detected in planned trajectory")
                return JointTrajectory([], [], [], success=False)
                
            # Create simple 3-point trajectory
            waypoints = [current_joints, pre_grasp_joints, final_joints]
            gripper_widths = [0.08, 0.08, grasp_pose.width]  # Open -> Open -> Close to target
            timestamps = [0.0, 2.0, 4.0]  # 4 second total approach (smooth but watchable)
            
            trajectory = JointTrajectory(
                joint_positions=waypoints,
                gripper_widths=gripper_widths,
                timestamps=timestamps,
                success=True
            )
            
            logger.info(f"Planned grasp trajectory with {len(waypoints)} waypoints")
            return trajectory
            
        except Exception as e:
            logger.error(f"Trajectory planning failed: {e}")
            return JointTrajectory([], [], [], success=False)
            
    def plan_lift_trajectory(self, grasp_joints: np.ndarray, lift_height: float = 0.15) -> JointTrajectory:
        """
        Plan trajectory to lift object after grasping.
        
        Args:
            grasp_joints: Joint configuration at grasp
            lift_height: How high to lift in meters
            
        Returns:
            Planned lift trajectory
        """
        try:
            # Get current end effector position
            fk_result = p.getLinkState(self.robot_id, self.end_effector_link)
            current_pos = np.array(fk_result[0])
            current_orn = np.array(fk_result[1])
            
            # Target position (lift up)
            lift_position = current_pos + np.array([0, 0, lift_height])
            
            # Solve IK for lift position
            lift_joints = p.calculateInverseKinematics(
                self.robot_id,
                self.end_effector_link,
                lift_position,
                current_orn,  # Keep same orientation
                lowerLimits=self.joint_limits_lower.tolist(),
                upperLimits=self.joint_limits_upper.tolist(),
                jointRanges=(self.joint_limits_upper - self.joint_limits_lower).tolist(),
                maxNumIterations=self.max_ik_iterations
            )
            lift_joints = np.array(lift_joints[:7])
            
            # Check collision
            if self.check_collision(lift_joints):
                logger.debug("Collision detected in lift trajectory")
                return JointTrajectory([], [], [], success=False)
                
            # Create lift trajectory
            waypoints = [grasp_joints, lift_joints]
            gripper_widths = [0.02, 0.02]  # Keep gripper closed
            timestamps = [0.0, 2.0]  # 2 second lift (smooth but watchable)
            
            trajectory = JointTrajectory(
                joint_positions=waypoints,
                gripper_widths=gripper_widths,
                timestamps=timestamps,
                success=True
            )
            
            logger.info(f"Planned lift trajectory to height {lift_height}m")
            return trajectory
            
        except Exception as e:
            logger.error(f"Lift planning failed: {e}")
            return JointTrajectory([], [], [], success=False)
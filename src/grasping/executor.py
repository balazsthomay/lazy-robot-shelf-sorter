"""
Grasp Execution with Feedback and Retry Logic

Handles grasp execution with quantified success detection,
force feedback, and intelligent retry strategies.
"""

import numpy as np
import pybullet as p
import time
from typing import List, Optional
from dataclasses import dataclass
import logging

from .planner import JointTrajectory, MotionPlanner
from .predictor import GraspPose

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    """Result of grasp execution attempt."""
    success: bool
    gripper_force: float           # Maximum force during grasp
    object_moved: bool            # Whether object was displaced
    lift_height_achieved: float   # Actual lift height achieved
    retry_count: int              # Number of retry attempts
    error_message: Optional[str] = None


class GraspExecutor:
    """
    Executes grasping operations with feedback and retry logic.
    
    Provides quantified success detection based on force feedback,
    object displacement, and retry strategies with pose variations.
    """
    
    def __init__(self, robot_id: int, motion_planner: MotionPlanner):
        """
        Initialize grasp executor.
        
        Args:
            robot_id: PyBullet robot body ID
            motion_planner: Motion planner instance
        """
        self.robot_id = robot_id
        self.motion_planner = motion_planner
        
        # Gripper joint indices (assuming last 2 joints)
        self.gripper_joints = [7, 8]
        
        # Success detection parameters
        self.min_grasp_force = 5.0        # Minimum force for successful grasp (N)
        self.min_object_displacement = 0.01  # Minimum object movement (m)
        self.lift_height_target = 0.10    # Target lift height (m)
        self.lift_height_tolerance = 0.05 # Acceptable lift height error (m)
        
        # Execution parameters
        self.max_retries = 3
        self.trajectory_step_time = 0.1   # Time per trajectory step
        
        logger.info("Initialized grasp executor")
        
    def _execute_trajectory(self, trajectory: JointTrajectory) -> bool:
        """
        Execute joint trajectory with gripper control.
        
        Args:
            trajectory: Planned joint trajectory
            
        Returns:
            True if trajectory executed successfully
        """
        try:
            for i, (joints, gripper_width, timestamp) in enumerate(zip(
                trajectory.joint_positions,
                trajectory.gripper_widths, 
                trajectory.timestamps
            )):
                # Set arm joint targets
                for joint_idx, target_pos in enumerate(joints):
                    p.setJointMotorControl2(
                        self.robot_id,
                        joint_idx,
                        p.POSITION_CONTROL,
                        targetPosition=target_pos,
                        force=240  # Max force for Franka Panda
                    )
                    
                # Set gripper targets
                for gripper_joint in self.gripper_joints:
                    p.setJointMotorControl2(
                        self.robot_id,
                        gripper_joint,
                        p.POSITION_CONTROL,
                        targetPosition=gripper_width/2,  # Each finger moves half the width
                        force=50
                    )
                    
                # Wait for motion to complete
                if i > 0:
                    time_step = trajectory.timestamps[i] - trajectory.timestamps[i-1]
                    steps = int(time_step / self.trajectory_step_time)
                    for _ in range(steps):
                        p.stepSimulation()
                        time.sleep(self.trajectory_step_time)
                        
            return True
            
        except Exception as e:
            logger.error(f"Trajectory execution failed: {e}")
            return False
            
    def _measure_gripper_force(self) -> float:
        """
        Measure current gripper force.
        
        Returns:
            Maximum force across gripper joints
        """
        try:
            forces = []
            for joint_idx in self.gripper_joints:
                joint_state = p.getJointState(self.robot_id, joint_idx)
                # Joint reaction forces are in joint_state[2] but need PyBullet Pro
                # For open source, we'll use a proxy based on joint torque
                joint_info = p.getJointInfo(self.robot_id, joint_idx)
                max_force = joint_info[10]  # Max force from URDF
                
                # Estimate force from applied torque (simplified)
                applied_force = abs(joint_state[3]) / max_force * 50  # Scale to realistic range
                forces.append(applied_force)
                
            return max(forces) if forces else 0.0
            
        except Exception as e:
            logger.debug(f"Force measurement failed: {e}")
            return 0.0
            
    def _detect_object_movement(self, 
                              object_ids: List[int],
                              initial_positions: List[np.ndarray],
                              threshold: float = 0.01) -> bool:
        """
        Detect if objects have moved significantly.
        
        Args:
            object_ids: List of object body IDs to track
            initial_positions: Initial positions of objects
            threshold: Movement threshold in meters
            
        Returns:
            True if any object moved beyond threshold
        """
        try:
            for obj_id, initial_pos in zip(object_ids, initial_positions):
                current_pos = np.array(p.getBasePositionAndOrientation(obj_id)[0])
                displacement = np.linalg.norm(current_pos - initial_pos)
                
                if displacement > threshold:
                    logger.debug(f"Object {obj_id} moved {displacement:.3f}m")
                    return True
                    
            return False
            
        except Exception as e:
            logger.debug(f"Object movement detection failed: {e}")
            return False
            
    def _get_object_ids_in_scene(self) -> List[int]:
        """
        Get all object IDs in the scene (excluding robot and ground).
        
        Returns:
            List of object body IDs
        """
        object_ids = []
        num_bodies = p.getNumBodies()
        
        for i in range(num_bodies):
            body_info = p.getBodyInfo(i)
            # Skip robot (self.robot_id) and ground (0)
            if i != self.robot_id and i != 0:
                object_ids.append(i)
                
        return object_ids
        
    def execute_grasp(self, grasp_pose: GraspPose) -> ExecutionResult:
        """
        Execute complete grasp sequence with success detection.
        
        Args:
            grasp_pose: Target grasp pose
            
        Returns:
            Execution result with success metrics
        """
        logger.info(f"Executing grasp at position {grasp_pose.position}")
        
        # Get initial object positions for movement detection
        object_ids = self._get_object_ids_in_scene()
        initial_positions = []
        for obj_id in object_ids:
            pos = np.array(p.getBasePositionAndOrientation(obj_id)[0])
            initial_positions.append(pos)
            
        # Plan approach trajectory
        approach_trajectory = self.motion_planner.plan_grasp_approach(grasp_pose)
        if not approach_trajectory.success:
            return ExecutionResult(
                success=False,
                gripper_force=0.0,
                object_moved=False,
                lift_height_achieved=0.0,
                retry_count=0,
                error_message="Approach trajectory planning failed"
            )
            
        # Execute approach
        if not self._execute_trajectory(approach_trajectory):
            return ExecutionResult(
                success=False,
                gripper_force=0.0,
                object_moved=False,
                lift_height_achieved=0.0,
                retry_count=0,
                error_message="Approach trajectory execution failed"
            )
            
        # Allow some settling time
        for _ in range(10):
            p.stepSimulation()
            time.sleep(0.01)
            
        # Measure grasp force
        max_force = self._measure_gripper_force()
        
        # Check object movement during grasp
        object_moved = self._detect_object_movement(
            object_ids, initial_positions, self.min_object_displacement
        )
        
        # Get current joint configuration for lift planning
        current_joints = []
        for i in range(7):  # Arm joints only
            joint_state = p.getJointState(self.robot_id, i)
            current_joints.append(joint_state[0])
        current_joints = np.array(current_joints)
        
        # Plan and execute lift
        lift_trajectory = self.motion_planner.plan_lift_trajectory(
            current_joints, self.lift_height_target
        )
        
        lift_height_achieved = 0.0
        if lift_trajectory.success:
            # Get initial end effector position
            initial_ef_state = p.getLinkState(self.robot_id, self.motion_planner.end_effector_link)
            initial_ef_height = initial_ef_state[0][2]
            
            # Execute lift
            if self._execute_trajectory(lift_trajectory):
                # Measure achieved lift height
                final_ef_state = p.getLinkState(self.robot_id, self.motion_planner.end_effector_link)
                final_ef_height = final_ef_state[0][2]
                lift_height_achieved = final_ef_height - initial_ef_height
                
        # Determine success based on multiple criteria
        force_success = max_force >= self.min_grasp_force
        lift_success = abs(lift_height_achieved - self.lift_height_target) < self.lift_height_tolerance
        
        # Combined success criteria (at least 2 out of 3)
        success_indicators = [force_success, object_moved, lift_success]
        overall_success = sum(success_indicators) >= 2
        
        logger.info(f"Grasp execution complete: force={max_force:.1f}N, "
                   f"moved={object_moved}, lift={lift_height_achieved:.3f}m, "
                   f"success={overall_success}")
        
        return ExecutionResult(
            success=overall_success,
            gripper_force=max_force,
            object_moved=object_moved,
            lift_height_achieved=lift_height_achieved,
            retry_count=0
        )
        
    def execute_with_retry(self, grasp_poses: List[GraspPose]) -> ExecutionResult:
        """
        Execute grasp with retry logic using multiple pose candidates.
        
        Args:
            grasp_poses: List of grasp pose candidates (sorted by confidence)
            
        Returns:
            Best execution result across all attempts
        """
        best_result = ExecutionResult(
            success=False,
            gripper_force=0.0,
            object_moved=False,
            lift_height_achieved=0.0,
            retry_count=0,
            error_message="No valid grasp poses provided"
        )
        
        if not grasp_poses:
            return best_result
            
        for retry_count, grasp_pose in enumerate(grasp_poses[:self.max_retries]):
            logger.info(f"Grasp attempt {retry_count + 1}/{min(len(grasp_poses), self.max_retries)} "
                       f"(confidence: {grasp_pose.confidence:.2f})")
                       
            result = self.execute_grasp(grasp_pose)
            result.retry_count = retry_count
            
            if result.success:
                logger.info(f"Grasp succeeded on attempt {retry_count + 1}")
                return result
                
            # Update best result if this attempt was better
            if (result.gripper_force > best_result.gripper_force or 
                result.lift_height_achieved > best_result.lift_height_achieved):
                best_result = result
                
        logger.warning(f"All grasp attempts failed after {min(len(grasp_poses), self.max_retries)} tries")
        return best_result
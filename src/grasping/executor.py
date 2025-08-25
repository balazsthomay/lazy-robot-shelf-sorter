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
class ExecutionMetrics:
    """Detailed execution metrics for Phase 4.3."""
    max_gripper_force: float      # Peak force during grasp (N)
    avg_gripper_force: float      # Average force during hold (N)
    object_displacement: float    # Total object movement (m)
    lift_height_achieved: float   # Actual lift height (m)
    grasp_stability_score: float  # 0-1 stability during lift
    execution_time: float         # Total execution time (s)
    contact_points: int           # Number of contact points detected
    force_consistency: float      # Force variation during hold (0-1)

@dataclass
class ExecutionResult:
    """Result of grasp execution attempt."""
    success: bool
    gripper_force: float           # Maximum force during grasp
    object_moved: bool            # Whether object was displaced
    lift_height_achieved: float   # Actual lift height achieved
    retry_count: int              # Number of retry attempts
    error_message: Optional[str] = None
    # Enhanced metrics for Phase 4.3
    metrics: Optional[ExecutionMetrics] = None


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
        
        # Enhanced metrics parameters for Phase 4.3
        self.force_measurement_samples = 50  # Samples during force measurement
        self.stability_threshold = 0.7       # Minimum stability score
        self.min_contact_points = 2          # Minimum contact points for success
        
        # Execution parameters
        self.max_retries = 3
        self.trajectory_step_time = 0.03  # Even smaller time steps for very smooth motion
        self.gui_observation_delay = 0.01  # Minimal delay
        
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
                        force=240,  # Max force for Franka Panda
                        maxVelocity=0.8  # Smooth, controlled movement
                    )
                    
                # Set gripper targets
                for gripper_joint in self.gripper_joints:
                    p.setJointMotorControl2(
                        self.robot_id,
                        gripper_joint,
                        p.POSITION_CONTROL,
                        targetPosition=gripper_width/2,  # Each finger moves half the width
                        force=50,
                        maxVelocity=0.3  # Slower gripper movement for gentle contact
                    )
                    
                # Smooth motion execution
                if i > 0:
                    time_step = trajectory.timestamps[i] - trajectory.timestamps[i-1]
                    steps = max(10, int(time_step / self.trajectory_step_time))  # Many small steps for smoothness
                    
                    # Interpolate smoothly between previous and current joint positions
                    prev_joints = trajectory.joint_positions[i-1]
                    curr_joints = trajectory.joint_positions[i]
                    prev_gripper = trajectory.gripper_widths[i-1]
                    curr_gripper = trajectory.gripper_widths[i]
                    
                    for step in range(steps):
                        # Linear interpolation
                        alpha = (step + 1) / steps
                        interp_joints = prev_joints + alpha * (curr_joints - prev_joints)
                        interp_gripper = prev_gripper + alpha * (curr_gripper - prev_gripper)
                        
                        # Set interpolated joint targets
                        for joint_idx, target_pos in enumerate(interp_joints):
                            p.setJointMotorControl2(
                                self.robot_id,
                                joint_idx,
                                p.POSITION_CONTROL,
                                targetPosition=target_pos,
                                force=240,
                                maxVelocity=0.8  # Smooth, controlled movement
                            )
                        
                        # Set interpolated gripper targets
                        for gripper_joint in self.gripper_joints:
                            p.setJointMotorControl2(
                                self.robot_id,
                                gripper_joint,
                                p.POSITION_CONTROL,
                                targetPosition=interp_gripper/2,
                                force=50,
                                maxVelocity=0.3  # Slower gripper movement for gentle contact
                            )
                        
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
    
    def _measure_detailed_force_metrics(self) -> tuple[float, float, float]:
        """
        Measure detailed force metrics over time.
        
        Returns:
            Tuple of (max_force, avg_force, force_consistency)
        """
        force_samples = []
        
        for _ in range(self.force_measurement_samples):
            force = self._measure_gripper_force()
            force_samples.append(force)
            p.stepSimulation()
            time.sleep(0.01)  # Small delay between samples
            
        if not force_samples:
            return 0.0, 0.0, 0.0
            
        max_force = max(force_samples)
        avg_force = np.mean(force_samples)
        force_std = np.std(force_samples)
        
        # Consistency: higher is better (lower variation)
        force_consistency = max(0.0, 1.0 - (force_std / max(avg_force, 1.0)))
        
        return max_force, avg_force, force_consistency
    
    def _measure_contact_points(self, object_ids: List[int]) -> int:
        """
        Count contact points between gripper and objects.
        
        Args:
            object_ids: List of object IDs to check contact with
            
        Returns:
            Total number of contact points
        """
        contact_count = 0
        
        for obj_id in object_ids:
            contacts = p.getContactPoints(bodyA=self.robot_id, bodyB=obj_id)
            # Filter for gripper contacts (joint indices 7 and 8)
            gripper_contacts = [
                c for c in contacts 
                if c[3] in self.gripper_joints or c[4] in self.gripper_joints
            ]
            contact_count += len(gripper_contacts)
            
        return contact_count
    
    def _calculate_stability_score(self, 
                                 initial_positions: List[np.ndarray],
                                 object_ids: List[int],
                                 num_samples: int = 30) -> float:
        """
        Calculate grasp stability during lift.
        
        Args:
            initial_positions: Initial object positions
            object_ids: Object IDs to track
            num_samples: Number of stability samples
            
        Returns:
            Stability score (0-1, higher is better)
        """
        if not object_ids:
            return 0.0
            
        position_variations = []
        
        for _ in range(num_samples):
            for i, obj_id in enumerate(object_ids):
                current_pos = np.array(p.getBasePositionAndOrientation(obj_id)[0])
                if i < len(initial_positions):
                    # Measure position variation relative to lift direction
                    xy_variation = np.linalg.norm(current_pos[:2] - initial_positions[i][:2])
                    position_variations.append(xy_variation)
            
            p.stepSimulation()
            time.sleep(0.02)
        
        if not position_variations:
            return 0.0
            
        # Lower variation = higher stability
        avg_variation = np.mean(position_variations)
        stability = max(0.0, 1.0 - (avg_variation / 0.05))  # Normalize to 0-1
        
        return min(1.0, stability)
            
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
        Execute complete grasp sequence with enhanced success detection.
        
        Args:
            grasp_pose: Target grasp pose
            
        Returns:
            Execution result with detailed metrics
        """
        start_time = time.time()
        logger.info(f"Executing grasp at position {grasp_pose.position}")
        print("   📍 Planning approach trajectory...")
        
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
        print("   🤖 Executing approach trajectory - watch robot move to grasp position")
        if not self._execute_trajectory(approach_trajectory):
            return ExecutionResult(
                success=False,
                gripper_force=0.0,
                object_moved=False,
                lift_height_achieved=0.0,
                retry_count=0,
                error_message="Approach trajectory execution failed"
            )
            
        # Allow some settling time for GUI observation
        print("   ✓ Approach complete - gripper closing on object")
        print("   🔍 Measuring grasp quality and contact stability...")
        for _ in range(20):
            p.stepSimulation()
            time.sleep(0.05)  # Slower settling for visual tracking
            
        # Measure detailed force metrics
        max_force, avg_force, force_consistency = self._measure_detailed_force_metrics()
        
        # Count contact points
        contact_points = self._measure_contact_points(object_ids)
        
        # Check object movement during grasp
        object_moved = self._detect_object_movement(
            object_ids, initial_positions, self.min_object_displacement
        )
        
        # Calculate object displacement magnitude
        total_displacement = 0.0
        if object_ids:
            for i, obj_id in enumerate(object_ids):
                if i < len(initial_positions):
                    current_pos = np.array(p.getBasePositionAndOrientation(obj_id)[0])
                    displacement = np.linalg.norm(current_pos - initial_positions[i])
                    total_displacement = max(total_displacement, displacement)
        
        # Get current joint configuration for lift planning
        current_joints = []
        for i in range(7):  # Arm joints only
            joint_state = p.getJointState(self.robot_id, i)
            current_joints.append(joint_state[0])
        current_joints = np.array(current_joints)
        
        # Plan and execute lift
        print("   ⬆️  Planning and executing lift trajectory")
        lift_trajectory = self.motion_planner.plan_lift_trajectory(
            current_joints, self.lift_height_target
        )
        
        lift_height_achieved = 0.0
        stability_score = 0.0
        
        if lift_trajectory.success:
            # Get initial end effector position
            initial_ef_state = p.getLinkState(self.robot_id, self.motion_planner.end_effector_link)
            initial_ef_height = initial_ef_state[0][2]
            
            # Execute lift
            print("   📊 Measuring stability during lift...")
            if self._execute_trajectory(lift_trajectory):
                # Measure achieved lift height
                final_ef_state = p.getLinkState(self.robot_id, self.motion_planner.end_effector_link)
                final_ef_height = final_ef_state[0][2]
                lift_height_achieved = final_ef_height - initial_ef_height
                
                # Measure stability during lift
                stability_score = self._calculate_stability_score(
                    initial_positions, object_ids
                )
                
        # Calculate execution time
        execution_time = time.time() - start_time
        
        # Create detailed metrics
        metrics = ExecutionMetrics(
            max_gripper_force=max_force,
            avg_gripper_force=avg_force,
            object_displacement=total_displacement,
            lift_height_achieved=lift_height_achieved,
            grasp_stability_score=stability_score,
            execution_time=execution_time,
            contact_points=contact_points,
            force_consistency=force_consistency
        )
        
        # Enhanced success criteria for Phase 4.3
        force_success = max_force >= self.min_grasp_force
        lift_success = abs(lift_height_achieved - self.lift_height_target) < self.lift_height_tolerance
        contact_success = contact_points >= self.min_contact_points
        stability_success = stability_score >= self.stability_threshold
        
        # Combined success criteria (at least 3 out of 5)
        success_indicators = [
            force_success, object_moved, lift_success, 
            contact_success, stability_success
        ]
        overall_success = sum(success_indicators) >= 3
        
        # Log detailed results
        logger.info(f"Grasp execution complete:")
        logger.info(f"  Force: {max_force:.1f}N (avg: {avg_force:.1f}N, consistency: {force_consistency:.2f})")
        logger.info(f"  Movement: {object_moved} (displacement: {total_displacement:.3f}m)")
        logger.info(f"  Lift: {lift_height_achieved:.3f}m (target: {self.lift_height_target:.3f}m)")
        logger.info(f"  Contacts: {contact_points} (min: {self.min_contact_points})")
        logger.info(f"  Stability: {stability_score:.2f} (min: {self.stability_threshold:.2f})")
        logger.info(f"  Success: {overall_success} ({sum(success_indicators)}/5 criteria met)")
        
        # Print user-friendly summary
        print(f"   📊 Execution Metrics:")
        print(f"      Force: {max_force:.1f}N | Movement: {'✓' if object_moved else '✗'} | Lift: {lift_height_achieved:.3f}m")
        print(f"      Contacts: {contact_points} | Stability: {stability_score:.2f} | Time: {execution_time:.1f}s")
        print(f"   {'🎯 SUCCESS' if overall_success else '❌ FAILED'}: {sum(success_indicators)}/5 criteria met")
        
        return ExecutionResult(
            success=overall_success,
            gripper_force=max_force,
            object_moved=object_moved,
            lift_height_achieved=lift_height_achieved,
            retry_count=0,
            metrics=metrics
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
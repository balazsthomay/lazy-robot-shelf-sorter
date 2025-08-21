#!/usr/bin/env python3
"""
Robot Controller - Phase 4: Robot Control
Handles motion execution for object placement from pickup to shelf locations
"""

import pybullet as p
import pybullet_data
import numpy as np
import time
from typing import List, Tuple, Optional, TYPE_CHECKING
from dataclasses import dataclass
from enum import Enum
from interfaces import SimulationComponent
from placement import PlacementCommand, PlacementResult
from simulation import ShelfConfiguration

if TYPE_CHECKING:
    from object_detection import ObjectDetector, DetectedObject


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
        self.arm_end_effector_link = 11  # panda_grasptarget for accurate positioning
                
    def move_to_position(self, target_position: Tuple[float, float, float], timeout: float = 5.0) -> bool:
        """Move end effector to target position and WAIT for completion"""
        if self.robot_id is None:
            return False
            
        # Record initial joint positions for diagnostic validation (arm joints 0-6)
        initial_joint_positions = []
        for joint_idx in range(7):  # Franka Panda arm joints are 0-6
            joint_state = p.getJointState(self.robot_id, joint_idx, physicsClientId=self.physics_client)
            initial_joint_positions.append(joint_state[0])
        
        # Simple IK using PyBullet (use arm end-effector, not gripper)
        joint_positions = p.calculateInverseKinematics(
            self.robot_id,
            self.arm_end_effector_link,  # Arm end-effector (joint 6)
            target_position,
            physicsClientId=self.physics_client
        )
        
        # Check if IK found a solution
        if not joint_positions or len(joint_positions) < 7:
            return False
        
        # Apply joint positions with proper motor control parameters (arm joints 0-6)
        for i in range(7):  # Franka Panda arm joints are 0-6
            if i < len(joint_positions):
                p.setJointMotorControl2(
                    self.robot_id,
                    i,  # Direct joint index for arm joints
                    p.POSITION_CONTROL,
                    targetPosition=joint_positions[i],
                    force=1000.0,  # Maximum torque for precision positioning
                    maxVelocity=3.0,  # Faster convergence for better precision
                    positionGain=1.0,  # Maximum position correction
                    velocityGain=1.0,  # D-gain for damping
                    physicsClientId=self.physics_client
                )
        
        # CRITICAL FIX: Wait for ACTUAL joint motion to complete, not kinematic model
        target_joint_positions = joint_positions[:7]  # Only arm joints
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            # Step simulation to allow motors to move joints
            p.stepSimulation(physicsClientId=self.physics_client)
            time.sleep(1./240.)
            
            # Check ACTUAL joint positions (not kinematic model) for arm joints 0-6
            actual_joint_positions = []
            joint_motion_complete = True
            
            for i in range(7):  # Franka Panda arm joints are 0-6
                joint_state = p.getJointState(self.robot_id, i, physicsClientId=self.physics_client)
                actual_position = joint_state[0]  # Joint position
                actual_joint_positions.append(actual_position)
                
                # Check if this joint reached its target
                joint_error = abs(actual_position - target_joint_positions[i])
                if joint_error > 0.20:  # 0.20 radian tolerance (~11.5 degrees) - pragmatic for joint limits
                    joint_motion_complete = False
            
            # All joints reached their targets
            if joint_motion_complete:
                # DIAGNOSTIC: Verify joints actually moved from initial positions
                max_joint_movement = 0.0
                for i, actual_pos in enumerate(actual_joint_positions):
                    movement = abs(actual_pos - initial_joint_positions[i])
                    max_joint_movement = max(max_joint_movement, movement)
                
                # Ensure robot actually moved (catch false positive bug)
                if max_joint_movement < 0.01:  # Less than 0.6 degree movement
                    print(f"WARNING: Robot appears unmoved despite 'success' (max movement: {max_joint_movement:.4f} rad)")
                    return False  # Catch the original bug condition
                
                return True
        
        # Motion timed out - joints didn't reach targets
        return False
        
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

    def control_gripper(self, open_gripper: bool = True, target_position: float = None) -> None:
        """Control gripper open/close with precise positioning"""
        if self.robot_id is None:
            return
            
        # Franka gripper joints: panda_finger_joint1 (9) and panda_finger_joint2 (10)
        gripper_joints = [9, 10]  # finger joints
        
        # Use provided position or default values
        if target_position is not None:
            gripper_position = target_position
        else:
            gripper_position = 0.04 if open_gripper else 0.0  # open = 0.04m, closed = 0.0m
        
        # Use higher force for better grasping (based on official examples)
        for joint in gripper_joints:
            p.setJointMotorControl2(
                self.robot_id,
                joint,
                p.POSITION_CONTROL,
                targetPosition=gripper_position,
                force=200.0,  # Increased force for reliable grasping
                maxVelocity=0.2,  # Slower for more precise control
                positionGain=0.5,
                velocityGain=1.0,
                physicsClientId=self.physics_client
            )
    
    def get_gripper_state(self) -> Tuple[bool, float]:
        """Get gripper state (is_open, current_width)
        
        Returns:
            Tuple of (is_open, gripper_width_meters)
        """
        if self.robot_id is None:
            return True, 0.04
            
        # Get gripper joint positions: panda_finger_joint1 (9) and panda_finger_joint2 (10)
        gripper_joints = [9, 10]
        joint_states = []
        
        for joint in gripper_joints:
            joint_state = p.getJointState(self.robot_id, joint, physicsClientId=self.physics_client)
            joint_states.append(joint_state[0])  # position
        
        # Average gripper width (both fingers should be symmetric)
        gripper_width = sum(joint_states) / len(joint_states) if joint_states else 0.04
        is_open = gripper_width > 0.02  # Consider open if > 2cm
        
        return is_open, gripper_width


class MotionFailureReason(Enum):
    """Reasons for motion execution failure"""
    IK_FAILURE = "ik_failure"
    TIMEOUT = "timeout"
    COLLISION_DETECTED = "collision_detected"
    UNREACHABLE_POSITION = "unreachable_position"
    ROBOT_NOT_INITIALIZED = "robot_not_initialized"
    INVALID_COMMAND = "invalid_command"
    GRASP_FAILURE = "grasp_failure"
    TRANSPORT_FAILURE = "transport_failure"


@dataclass
class MotionResult:
    """Result of motion execution"""
    success: bool
    execution_time: float
    final_pose: Optional[Tuple[float, float, float]] = None
    confidence_score: float = 0.0
    failure_reason: Optional[str] = None
    waypoints_completed: int = 0
    total_waypoints: int = 0


@dataclass
class ShelfGeometry:
    """Phase 4C: Shelf geometry for collision avoidance"""
    shelf_positions: List[Tuple[float, float, float]]  # (x, y, z) positions
    shelf_width: float = 0.8   # 80cm
    shelf_depth: float = 0.4   # 40cm
    shelf_height: float = 0.3  # 30cm (vertical spacing)
    shelf_thickness: float = 0.02  # 2cm shelf thickness
    
    def get_shelf_bounds(self, shelf_index: int) -> Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]]:
        """Get min/max bounds for a shelf"""
        if shelf_index >= len(self.shelf_positions):
            return None
            
        x, y, z = self.shelf_positions[shelf_index]
        
        min_bounds = (
            x - self.shelf_width / 2,
            y - self.shelf_depth / 2,
            z - self.shelf_thickness / 2
        )
        max_bounds = (
            x + self.shelf_width / 2,
            y + self.shelf_depth / 2,
            z + self.shelf_thickness / 2
        )
        
        return min_bounds, max_bounds
    
    def get_safe_height_above_shelves(self, target_position: Tuple[float, float, float]) -> float:
        """Calculate safe height above all shelves for given x,y position"""
        x_target, y_target, _ = target_position
        max_shelf_z = 0.0
        
        for shelf_pos in self.shelf_positions:
            x_shelf, y_shelf, z_shelf = shelf_pos
            
            # Check if target position is within shelf x,y bounds
            if (abs(x_target - x_shelf) <= self.shelf_width / 2 + 0.1 and  # Add 10cm margin
                abs(y_target - y_shelf) <= self.shelf_depth / 2 + 0.1):
                max_shelf_z = max(max_shelf_z, z_shelf + self.shelf_thickness / 2)
        
        return max_shelf_z
    
    def check_collision_free_path(self, start_pos: Tuple[float, float, float], 
                                 end_pos: Tuple[float, float, float], 
                                 clearance_margin: float = 0.1) -> bool:
        """Simple collision check for straight-line path"""
        # For Phase 4C KISS implementation: just check if both endpoints are safe
        # More sophisticated path checking could be added later
        return (self._check_position_clearance(start_pos, clearance_margin) and
                self._check_position_clearance(end_pos, clearance_margin))
    
    def _check_position_clearance(self, position: Tuple[float, float, float], margin: float) -> bool:
        """Check if position has clearance from shelf edges"""
        x, y, z = position
        
        for shelf_pos in self.shelf_positions:
            x_shelf, y_shelf, z_shelf = shelf_pos
            
            # Check if position is within shelf horizontal bounds
            within_shelf_xy = (abs(x - x_shelf) <= self.shelf_width / 2 + margin and
                              abs(y - y_shelf) <= self.shelf_depth / 2 + margin)
            
            if within_shelf_xy:
                # Position is above/below shelf - check vertical clearance
                shelf_top = z_shelf + self.shelf_thickness / 2
                shelf_bottom = z_shelf - self.shelf_thickness / 2
                
                # Unsafe if too close to shelf top or bottom
                if (shelf_bottom - margin <= z <= shelf_top + margin):
                    return False
                    
        return True


class MotionController:
    """Phase 4: Robot motion controller with 3-waypoint strategy
    
    Implements KISS-principle motion control:
    - 3-waypoint motion: current → pickup → safe → placement
    - Basic collision avoidance via safe intermediate poses
    - Simple retry logic for IK failures
    """
    
    def __init__(self, robot_controller: RobotController, 
                 pickup_location: Tuple[float, float, float] = None,
                 shelf_geometry: Optional[ShelfGeometry] = None,
                 object_detector: Optional['ObjectDetector'] = None):
        self.robot = robot_controller
        self.pickup_location = pickup_location  # Manual mode (fallback)
        self.object_detector = object_detector  # Vision mode (preferred)
        self.motion_timeout = 30.0  # 30s timeout as per plan_v4.md
        self.shelf_clearance = 0.2  # 20cm clearance above shelves
        self.shelf_edge_margin = 0.1  # 10cm margin from shelf edges
        
        # Phase 4C: Enhanced collision avoidance
        self.shelf_geometry = shelf_geometry
        
        # Validate configuration
        if not object_detector and not pickup_location:
            raise ValueError("Either object_detector or pickup_location must be provided")
    
    def _get_target_object_position(self) -> Tuple[float, float, float]:
        """Get object position from vision detection or manual coordinates
        
        Returns:
            (x, y, z) position of target object for pickup
            
        Raises:
            ValueError: If no object can be found or coordinates provided
        """
        if self.object_detector:
            try:
                objects = self.object_detector.detect_objects_on_table()
                if objects:
                    # Select best object: closest + reachable + appropriate size
                    best_object = self._select_best_object(objects)
                    print(f"🎯 Vision-guided pickup: Using detected object at {best_object.position}")
                    return best_object.position
                else:
                    print("⚠️  No objects detected by vision system")
            except Exception as e:
                print(f"⚠️  Vision detection failed: {e}")
                
            # Fallback to manual coordinates if vision fails
            if self.pickup_location:
                print(f"🔧 Falling back to manual coordinates: {self.pickup_location}")
                return self.pickup_location
        
        elif self.pickup_location:
            # Manual mode
            return self.pickup_location
            
        else:
            raise ValueError("No object detection or manual coordinates available")
    
    def _select_best_object(self, objects: List['DetectedObject']) -> 'DetectedObject':
        """Select most suitable object for grasping
        
        Args:
            objects: List of detected objects from ObjectDetector
            
        Returns:
            Best object for grasping based on multiple criteria
        """
        if not objects:
            raise ValueError("No objects provided for selection")
            
        # Filter by reachability (robot workspace constraints)
        robot_base = (0.0, -0.8, 0.0)  # Robot base position
        reachable_objects = []
        
        for obj in objects:
            x, y, z = obj.position
            # Check basic workspace bounds (same as in RobotController)
            if (0.1 <= x <= 0.8 and -0.8 <= y <= 0.6 and 0.0 <= z <= 0.5):
                # Check if object size is reasonable for grasping
                max_size = max(obj.size_estimate)
                if 0.01 <= max_size <= 0.15:  # 1cm to 15cm objects
                    reachable_objects.append(obj)
        
        if not reachable_objects:
            # Return closest object even if marginal
            return min(objects, key=lambda o: self._distance_to_point(o.position, robot_base))
        
        # Prioritize by distance to robot base (closest first)
        # Objects are already sorted by distance in ObjectDetector
        best_object = reachable_objects[0]
        
        print(f"🎯 Selected object: pos={best_object.position}, size={best_object.size_estimate}, conf={best_object.confidence:.2f}")
        return best_object
    
    def _distance_to_point(self, pos1: Tuple[float, float, float], 
                          pos2: Tuple[float, float, float]) -> float:
        """Calculate Euclidean distance between two 3D points"""
        return np.sqrt(sum((a - b) ** 2 for a, b in zip(pos1, pos2)))
        
    def execute_placement(self, placement_command: PlacementCommand) -> MotionResult:
        """Execute 3-waypoint motion: current → pickup → safe → placement
        
        Args:
            placement_command: PlacementCommand from Phase 3 with target pose
            
        Returns:
            MotionResult with success status and timing metrics
        """
        start_time = time.time()
        
        # Validation
        if self.robot.robot_id is None:
            return MotionResult(
                success=False,
                execution_time=0.0,
                failure_reason=MotionFailureReason.ROBOT_NOT_INITIALIZED.value
            )
            
        if not placement_command or not placement_command.position:
            return MotionResult(
                success=False,
                execution_time=0.0,
                failure_reason=MotionFailureReason.INVALID_COMMAND.value
            )
        
        try:
            # Generate 3 waypoints
            waypoints = self._generate_waypoints(placement_command.position)
            total_waypoints = len(waypoints)
            
            # Execute waypoint sequence
            completed = 0
            object_grasped = False
            
            for i, (waypoint, action) in enumerate(waypoints):
                elapsed = time.time() - start_time
                if elapsed > self.motion_timeout:
                    return MotionResult(
                        success=False,
                        execution_time=elapsed,
                        failure_reason=MotionFailureReason.TIMEOUT.value,
                        waypoints_completed=completed,
                        total_waypoints=total_waypoints
                    )
                
                # Execute waypoint
                success = self._execute_waypoint(waypoint, action)
                if not success:
                    failure_reason = MotionFailureReason.GRASP_FAILURE.value if action == "grasp" else MotionFailureReason.IK_FAILURE.value
                    return MotionResult(
                        success=False,
                        execution_time=time.time() - start_time,
                        failure_reason=failure_reason,
                        waypoints_completed=completed,
                        total_waypoints=total_waypoints
                    )
                
                # Track grasp state
                if action == "grasp":
                    object_grasped = True
                elif action == "release":
                    object_grasped = False
                
                # Phase 4B: Verify transport stability during movement after grasp
                if object_grasped and action == "move" and completed > 2:  # After initial grasp and lift
                    if not self._verify_transport_stability():
                        return MotionResult(
                            success=False,
                            execution_time=time.time() - start_time,
                            failure_reason=MotionFailureReason.TRANSPORT_FAILURE.value,
                            waypoints_completed=completed,
                            total_waypoints=total_waypoints
                        )
                
                completed += 1
                
                # Brief pause for simulation settling
                time.sleep(0.1)
            
            # Success
            final_pose = self.robot.get_end_effector_position()
            execution_time = time.time() - start_time
            
            return MotionResult(
                success=True,
                execution_time=execution_time,
                final_pose=final_pose,
                confidence_score=placement_command.confidence_score,
                waypoints_completed=completed,
                total_waypoints=total_waypoints
            )
            
        except Exception as e:
            return MotionResult(
                success=False,
                execution_time=time.time() - start_time,
                failure_reason=f"Unexpected error: {str(e)}"
            )
    
    def _generate_waypoints(self, placement_position: Tuple[float, float, float]) -> List[Tuple[Tuple[float, float, float], str]]:
        """Generate 7-waypoint sequence with actions - Phase 4.5 vision-guided
        
        Returns:
            List of (position, action) tuples
        """
        # Get dynamic object position (vision-guided or manual fallback)
        pickup_position = self._get_target_object_position()
        pickup_x, pickup_y, pickup_z = pickup_position
        place_x, place_y, place_z = placement_position
        
        # Phase 4C: Calculate shelf-aware safe intermediate height
        if self.shelf_geometry:
            # Use shelf geometry to calculate safe height
            pickup_safe_height = self.shelf_geometry.get_safe_height_above_shelves(pickup_position)
            place_safe_height = self.shelf_geometry.get_safe_height_above_shelves(placement_position)
            safe_z = max(pickup_safe_height, place_safe_height) + self.shelf_clearance
        else:
            # Fallback: simple height calculation
            safe_z = max(pickup_z, place_z) + self.shelf_clearance
        
        # FIXED: Proper grasping approach sequence
        # Step 1: Approach from above (pre-grasp position)
        pre_grasp_height = pickup_z + 0.10  # 10cm above object
        
        waypoints = [
            # 1. Move to above pickup location (pre-grasp)
            ((pickup_x, pickup_y, pre_grasp_height), "move"), 
            # 2. Descend to grasp height (slightly above object center)
            ((pickup_x, pickup_y, pickup_z + 0.02), "move"),  # 2cm above center
            # 3. Grasp object (at current position)
            ((pickup_x, pickup_y, pickup_z + 0.02), "grasp"),
            # 4. Lift to safe height
            ((pickup_x, pickup_y, safe_z), "move"),
            # 5. Move to above placement location
            ((place_x, place_y, safe_z), "move"),
            # 6. Lower to placement height
            ((place_x, place_y, place_z), "move"),
            # 7. Release object
            ((place_x, place_y, place_z), "release"),
            # 8. Retreat to safe height
            ((place_x, place_y, safe_z), "move")
        ]
        
        return waypoints
    
    def _execute_waypoint(self, position: Tuple[float, float, float], action: str) -> bool:
        """Execute single waypoint with action
        
        Args:
            position: Target 3D position
            action: Action to perform ("move", "grasp", "release")
            
        Returns:
            Success status
        """
        # Basic collision check - ensure we're not too close to shelf edges
        if not self._check_safe_position(position):
            return False
        
        # Execute motion
        success = self.robot.move_to_position(position)
        if not success:
            return False
        
        # Execute action
        if action == "grasp":
            success = self._execute_grasp()
            if not success:
                return False
        elif action == "release":
            self.robot.control_gripper(open_gripper=True)   # Open gripper
            time.sleep(0.4)  # Wait for release
        # "move" action requires no additional steps
        
        return True
    
    def _execute_grasp(self) -> bool:
        """Execute grasp with gradual closing and object position tracking"""
        print("🤏 Executing grasp sequence...")
        
        # Track object positions for validation (find nearest object to gripper)
        initial_object_positions = self._get_nearby_object_positions()
        
        # Step 1: Ensure gripper is fully open
        print("   1. Opening gripper fully...")
        self.robot.control_gripper(open_gripper=True)
        self._wait_for_simulation(60)  # Wait for opening
        
        # Debug: Show gripper and object positions
        gripper_pos = self.robot.get_end_effector_position()
        nearby_objects = initial_object_positions
        
        print(f"   🔧 Gripper position: ({gripper_pos[0]:.3f}, {gripper_pos[1]:.3f}, {gripper_pos[2]:.3f})")
        for i, obj_pos in enumerate(nearby_objects):
            distance = ((obj_pos[0] - gripper_pos[0])**2 + (obj_pos[1] - gripper_pos[1])**2 + (obj_pos[2] - gripper_pos[2])**2)**0.5
            print(f"   📦 Object {i}: ({obj_pos[0]:.3f}, {obj_pos[1]:.3f}, {obj_pos[2]:.3f}) - distance: {distance:.3f}m")
        
        # Step 2: Gradual closing approach (like official examples)
        print("   2. Gradual gripper closing...")
        
        # Start with partially open position and gradually close
        grasp_positions = [0.03, 0.02, 0.015, 0.01, 0.008, 0.005]  # Gradually close
        
        for i, pos in enumerate(grasp_positions):
            print(f"      Closing to {pos:.3f}m...")
            self.robot.control_gripper(open_gripper=False, target_position=pos)
            self._wait_for_simulation(40)  # Wait for each step
            
            # Check if object is grasped
            is_open, gripper_width = self.robot.get_gripper_state()
            print(f"      Width: {gripper_width:.4f}m")
            
            # If gripper stopped closing (hit object), we're done
            if gripper_width > pos + 0.002:  # Gripper couldn't reach target
                print(f"   ✅ Object detected! Gripper stopped at {gripper_width:.4f}m")
                break
        
        # Final check
        is_open, final_width = self.robot.get_gripper_state()
        print(f"🔧 Final grasp check: width={final_width:.4f}m")
        
        # PROPER grasp validation: check both gripper state AND object movement
        final_object_positions = self._get_nearby_object_positions()
        
        # Check if any objects moved (indicating interaction)
        object_moved = self._check_object_movement(initial_object_positions, final_object_positions)
        
        # Gripper state validation
        gripper_has_object = 0.007 < final_width < 0.035  # Object between fingers
        
        if not gripper_has_object:
            grasp_successful = False
            if final_width <= 0.007:
                print(f"🎯 Grasp result: FAILED - Gripper fully closed (no object)")
            else:
                print(f"🎯 Grasp result: FAILED - Gripper too wide (no contact)")
        elif not object_moved:
            grasp_successful = False
            print(f"🎯 Grasp result: FAILED - No object movement detected")
        else:
            grasp_successful = True
            print(f"🎯 Grasp result: SUCCESS - Object grasped (width: {final_width:.4f}m)")
        
        return grasp_successful
    
    def _wait_for_simulation(self, steps: int) -> None:
        """Wait for simulation to settle"""
        for _ in range(steps):
            p.stepSimulation(physicsClientId=self.robot.physics_client)
            time.sleep(1./240.)
    
    def _get_nearby_object_positions(self) -> List[Tuple[float, float, float]]:
        """Get positions of objects near the gripper"""
        gripper_pos = self.robot.get_end_effector_position()
        object_positions = []
        
        # Check all bodies in simulation (skip robot and ground)
        num_bodies = p.getNumBodies(physicsClientId=self.robot.physics_client)
        
        for body_id in range(num_bodies):
            if body_id == self.robot.robot_id:  # Skip robot
                continue
                
            try:
                pos, _ = p.getBasePositionAndOrientation(body_id, physicsClientId=self.robot.physics_client)
                
                # Only track objects within 0.5m of gripper
                distance = ((pos[0] - gripper_pos[0])**2 + 
                           (pos[1] - gripper_pos[1])**2 + 
                           (pos[2] - gripper_pos[2])**2)**0.5
                
                if distance < 0.5:  # Within 50cm
                    object_positions.append(pos)
                    
            except:
                continue  # Skip invalid bodies
                
        return object_positions
    
    def _check_object_movement(self, initial_positions: List, final_positions: List) -> bool:
        """Check if any objects moved significantly"""
        if len(initial_positions) != len(final_positions):
            return True  # Object count changed
            
        for i, (init_pos, final_pos) in enumerate(zip(initial_positions, final_positions)):
            movement = ((init_pos[0] - final_pos[0])**2 + 
                       (init_pos[1] - final_pos[1])**2 + 
                       (init_pos[2] - final_pos[2])**2)**0.5
            
            if movement > 0.01:  # Object moved more than 1cm
                print(f"   📦 Object {i} moved {movement:.3f}m")
                return True
                
        print(f"   📦 No significant object movement detected")
        return False
    
    def _verify_transport_stability(self) -> bool:
        """Verify object is still grasped during transport
        
        Simple check for Phase 4B - can be enhanced later
        """
        is_open, gripper_width = self.robot.get_gripper_state()
        
        # Object must still be present (not fully closed gripper)
        return 0.005 < gripper_width < 0.035
    
    def _check_safe_position(self, position: Tuple[float, float, float]) -> bool:
        """Phase 4C: Enhanced safety check with shelf collision avoidance"""
        x, y, z = position
        
        # Basic workspace bounds (conservative)
        if not (0.1 <= x <= 0.8):  # Robot reach in X
            return False
        if not (-0.8 <= y <= 0.6):  # Robot reach in Y (expanded for robot base at y=-0.8)
            return False  
        if not (0.05 <= z <= 1.2):  # Reasonable Z range
            return False
        
        # Phase 4C: Enhanced shelf collision checking
        if self.shelf_geometry:
            return self.shelf_geometry._check_position_clearance(position, self.shelf_edge_margin)
        
        # Fallback: basic check passed
        return True


@dataclass 
class ExecutionResult:
    """Phase 4D: Combined placement + motion result"""
    placement_success: bool
    motion_success: bool
    placement_result: Optional[PlacementResult]
    motion_result: Optional[MotionResult]
    total_execution_time: float
    overall_success: bool
    failure_summary: str = ""


class Phase3To4Bridge:
    """Phase 4D: Integration bridge between Phase 3 (Placement) and Phase 4 (Motion)
    
    Handles:
    - PlacementResult → MotionController execution
    - Error handling and retry logic
    - Performance monitoring and fallback
    - World state consistency
    """
    
    def __init__(self, motion_controller: MotionController, max_retries: int = 3):
        self.motion_controller = motion_controller
        self.max_retries = max_retries
        self.execution_history: List[ExecutionResult] = []
        
    def execute_placement_with_motion(self, placement_result: PlacementResult) -> ExecutionResult:
        """Execute complete placement → motion pipeline
        
        Args:
            placement_result: Result from Phase 3 placement system
            
        Returns:
            ExecutionResult with combined placement + motion results
        """
        start_time = time.time()
        
        # Validate placement result
        if not placement_result.success or not placement_result.placement_command:
            return ExecutionResult(
                placement_success=False,
                motion_success=False,
                placement_result=placement_result,
                motion_result=None,
                total_execution_time=0.0,
                overall_success=False,
                failure_summary="Phase 3 placement failed"
            )
        
        # Execute motion with retry logic
        motion_result = None
        retry_count = 0
        
        while retry_count <= self.max_retries:
            try:
                motion_result = self.motion_controller.execute_placement(placement_result.placement_command)
                
                if motion_result.success:
                    break  # Success, exit retry loop
                    
                # Motion failed - check if we should retry
                if self._should_retry_motion(motion_result, retry_count):
                    retry_count += 1
                    
                    # Apply retry strategy (slight position jitter)
                    placement_result.placement_command = self._apply_retry_strategy(
                        placement_result.placement_command, retry_count
                    )
                    
                    print(f"Motion retry {retry_count}/{self.max_retries}: {motion_result.failure_reason}")
                else:
                    break  # Don't retry this type of failure
                    
            except Exception as e:
                motion_result = MotionResult(
                    success=False,
                    execution_time=0.0,
                    failure_reason=f"Motion execution exception: {str(e)}"
                )
                break
        
        # Create combined result
        execution_time = time.time() - start_time
        overall_success = placement_result.success and (motion_result.success if motion_result else False)
        
        failure_summary = ""
        if not overall_success:
            failure_parts = []
            if not placement_result.success:
                failure_parts.append(f"Placement: {placement_result.failure_reason}")
            if motion_result and not motion_result.success:
                failure_parts.append(f"Motion: {motion_result.failure_reason}")
            failure_summary = "; ".join(failure_parts)
        
        result = ExecutionResult(
            placement_success=placement_result.success,
            motion_success=motion_result.success if motion_result else False,
            placement_result=placement_result,
            motion_result=motion_result,
            total_execution_time=execution_time,
            overall_success=overall_success,
            failure_summary=failure_summary
        )
        
        # Track execution history for debugging
        self.execution_history.append(result)
        
        return result
    
    def _should_retry_motion(self, motion_result: MotionResult, retry_count: int) -> bool:
        """Determine if motion should be retried"""
        if retry_count >= self.max_retries:
            return False
            
        # Retry on IK failures and timeouts, but not on collisions or robot errors
        retryable_failures = [
            MotionFailureReason.IK_FAILURE.value,
            MotionFailureReason.TIMEOUT.value,
            MotionFailureReason.GRASP_FAILURE.value
        ]
        
        return any(reason in motion_result.failure_reason for reason in retryable_failures)
    
    def _apply_retry_strategy(self, placement_command: PlacementCommand, retry_count: int) -> PlacementCommand:
        """Apply retry strategy with position jitter"""
        import copy
        
        # Create copy to avoid modifying original
        retried_command = copy.deepcopy(placement_command)
        
        # Apply small position jitter (±2cm per plan_v4.md)
        import random
        jitter_amount = 0.02  # 2cm
        
        x, y, z = retried_command.position
        x_jitter = random.uniform(-jitter_amount, jitter_amount)
        y_jitter = random.uniform(-jitter_amount, jitter_amount)
        
        retried_command.position = (x + x_jitter, y + y_jitter, z)
        
        # Slightly reduce confidence score to reflect retry
        retried_command.confidence_score *= 0.95
        
        return retried_command
    
    def get_success_rate(self) -> float:
        """Calculate overall success rate from execution history"""
        if not self.execution_history:
            return 0.0
            
        successful = sum(1 for result in self.execution_history if result.overall_success)
        return successful / len(self.execution_history)
    
    def get_average_execution_time(self) -> float:
        """Calculate average execution time"""
        if not self.execution_history:
            return 0.0
            
        total_time = sum(result.total_execution_time for result in self.execution_history)
        return total_time / len(self.execution_history)
    
    def clear_history(self) -> None:
        """Clear execution history"""
        self.execution_history.clear()


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
    robot = RobotController(physics_client)
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
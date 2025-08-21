#!/usr/bin/env python3
"""
Phase 4A Test: Core MotionController + PyBullet IK integration
Tests the basic motion controller implementation
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pybullet as p
import pybullet_data
import time
from control import RobotController, MotionController, MotionResult
from placement import PlacementCommand


def test_motion_controller_basic():
    """Test basic MotionController functionality"""
    print("🧪 Testing Phase 4A: Core MotionController...")
    
    # Initialize PyBullet
    physics_client = p.connect(p.DIRECT)
    p.setGravity(0, 0, -9.81)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF("plane.urdf")
    
    try:
        # Initialize robot
        robot = RobotController(physics_client)
        robot.initialize()
        
        # Initialize motion controller
        pickup_location = (0.5, -0.5, 0.1)
        motion_controller = MotionController(robot, pickup_location)
        
        # Test placement command
        placement_cmd = PlacementCommand(
            object_id="test_object",
            zone_id="test_zone",
            position=(0.3, 0.2, 0.3),  # Shelf-like position
            orientation=(0, 0, 0, 1),  # Identity quaternion
            confidence_score=0.85
        )
        
        print(f"✅ Robot initialized with {robot.num_joints} joints")
        print(f"✅ Motion controller initialized")
        print(f"✅ Pickup location: {pickup_location}")
        print(f"✅ Target placement: {placement_cmd.position}")
        
        # Execute motion
        start_time = time.time()
        result = motion_controller.execute_placement(placement_cmd)
        execution_time = time.time() - start_time
        
        # Validate results
        print(f"\n📊 Motion Execution Results:")
        print(f"   Success: {result.success}")
        print(f"   Execution time: {result.execution_time:.2f}s")
        print(f"   Waypoints completed: {result.waypoints_completed}/{result.total_waypoints}")
        
        if result.success:
            print(f"   Final pose: {result.final_pose}")
            print(f"   Confidence: {result.confidence_score}")
        else:
            print(f"   Failure reason: {result.failure_reason}")
        
        # Success criteria
        success = (
            result.success and
            result.execution_time < 30.0 and  # <30s requirement
            result.waypoints_completed == result.total_waypoints and
            result.final_pose is not None
        )
        
        print(f"\n🎯 Phase 4A Test: {'PASSED' if success else 'FAILED'}")
        return success
        
    finally:
        robot.cleanup()
        p.disconnect()


def test_motion_controller_error_handling():
    """Test error handling in MotionController"""
    print("\n🧪 Testing MotionController error handling...")
    
    # Test with uninitialized robot
    robot = RobotController()  # Not initialized
    motion_controller = MotionController(robot)
    
    placement_cmd = PlacementCommand(
        object_id="test",
        zone_id="test",
        position=(0.3, 0.2, 0.3),
        orientation=(0, 0, 0, 1),
        confidence_score=0.5
    )
    
    result = motion_controller.execute_placement(placement_cmd)
    
    expected_failure = not result.success and "robot_not_initialized" in result.failure_reason
    print(f"✅ Error handling test: {'PASSED' if expected_failure else 'FAILED'}")
    
    return expected_failure


def test_waypoint_generation():
    """Test waypoint generation logic"""
    print("\n🧪 Testing waypoint generation...")
    
    robot = RobotController()  # Mock robot for testing
    pickup_location = (0.5, -0.5, 0.1)
    motion_controller = MotionController(robot, pickup_location)
    
    placement_position = (0.3, 0.2, 0.3)
    waypoints = motion_controller._generate_waypoints(placement_position)
    
    # Validate waypoint structure
    expected_actions = ["move", "grasp", "move", "move", "move", "release", "move"]
    actual_actions = [action for _, action in waypoints]
    
    actions_correct = actual_actions == expected_actions
    waypoint_count_correct = len(waypoints) == 7
    
    print(f"   Generated waypoints: {len(waypoints)}")
    print(f"   Actions: {actual_actions}")
    print(f"   Actions correct: {actions_correct}")
    print(f"   Count correct: {waypoint_count_correct}")
    
    # Check safe height calculation
    pickup_z = pickup_location[2]
    place_z = placement_position[2]
    expected_safe_z = max(pickup_z, place_z) + 0.2  # shelf_clearance
    
    safe_waypoints = [pos for pos, action in waypoints if pos[2] > max(pickup_z, place_z)]
    safe_height_correct = len(safe_waypoints) > 0
    
    print(f"   Safe height waypoints: {len(safe_waypoints)}")
    
    success = actions_correct and waypoint_count_correct and safe_height_correct
    print(f"✅ Waypoint generation test: {'PASSED' if success else 'FAILED'}")
    
    return success


def main():
    """Run Phase 4A tests"""
    print("🚀 Phase 4A Testing: Core MotionController + PyBullet IK Integration")
    print("=" * 60)
    
    tests = [
        test_waypoint_generation,
        test_motion_controller_error_handling,
        test_motion_controller_basic,
    ]
    
    results = []
    for test in tests:
        result = test()
        results.append(result)
        print()
    
    passed = sum(results)
    total = len(results)
    
    print("=" * 60)
    print(f"📊 Phase 4A Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 Phase 4A: Core MotionController implementation COMPLETE!")
        print("✅ Ready for Phase 4B: Grasp & Transport implementation")
    else:
        print("❌ Phase 4A: Some tests failed - review implementation")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
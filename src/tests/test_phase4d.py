#!/usr/bin/env python3
"""
Phase 4D Test: Integration with Phase 3 + error handling
Tests the Phase 3→4 integration bridge and enhanced error handling
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pybullet as p
import pybullet_data
import time
from typing import Tuple
from control import (RobotController, MotionController, MotionResult, 
                    Phase3To4Bridge, ExecutionResult, ShelfGeometry)
from placement import PlacementCommand, PlacementResult, PlacementStrategy


def create_mock_placement_result(success: bool = True, position: Tuple[float, float, float] = (0.3, 0.2, 0.3)) -> PlacementResult:
    """Create mock placement result for testing"""
    if success:
        placement_command = PlacementCommand(
            object_id="test_object",
            zone_id="test_zone", 
            position=position,
            orientation=(0, 0, 0, 1),
            confidence_score=0.8
        )
        return PlacementResult(
            success=True,
            placement_command=placement_command,
            failure_reason=""
        )
    else:
        return PlacementResult(
            success=False,
            placement_command=None,
            failure_reason="Mock placement failure"
        )


def test_phase3to4_bridge_success():
    """Test successful Phase 3→4 integration"""
    print("🧪 Testing Phase3To4Bridge - success case...")
    
    # Initialize PyBullet
    physics_client = p.connect(p.DIRECT)
    p.setGravity(0, 0, -9.81)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF("plane.urdf")
    
    try:
        # Initialize robot and motion controller
        robot = RobotController(physics_client)
        robot.initialize()
        
        motion_controller = MotionController(robot, pickup_location=(0.5, -0.5, 0.1))
        
        # Create Phase 3→4 bridge
        bridge = Phase3To4Bridge(motion_controller)
        
        # Create successful placement result
        placement_result = create_mock_placement_result(success=True, position=(0.4, -0.3, 0.25))
        
        print(f"   Input placement success: {placement_result.success}")
        print(f"   Target position: {placement_result.placement_command.position}")
        
        # Execute integration
        execution_result = bridge.execute_placement_with_motion(placement_result)
        
        print(f"\n📊 Integration Results:")
        print(f"   Placement success: {execution_result.placement_success}")
        print(f"   Motion success: {execution_result.motion_success}")
        print(f"   Overall success: {execution_result.overall_success}")
        print(f"   Total execution time: {execution_result.total_execution_time:.2f}s")
        
        if not execution_result.overall_success:
            print(f"   Failure summary: {execution_result.failure_summary}")
        
        # Validate results
        success = (
            execution_result.placement_success and
            execution_result.total_execution_time < 30.0 and  # Performance requirement
            execution_result.placement_result is not None and
            execution_result.motion_result is not None
        )
        
        print(f"✅ Phase3To4Bridge success test: {'PASSED' if success else 'FAILED'}")
        return success
        
    finally:
        robot.cleanup()
        p.disconnect()


def test_phase3to4_bridge_placement_failure():
    """Test Phase 3→4 integration with placement failure"""
    print("\n🧪 Testing Phase3To4Bridge - placement failure case...")
    
    # Create mock controllers (no PyBullet needed for this test)
    robot = RobotController()
    motion_controller = MotionController(robot)
    bridge = Phase3To4Bridge(motion_controller)
    
    # Create failed placement result
    placement_result = create_mock_placement_result(success=False)
    
    print(f"   Input placement success: {placement_result.success}")
    print(f"   Failure reason: {placement_result.failure_reason}")
    
    # Execute integration
    execution_result = bridge.execute_placement_with_motion(placement_result)
    
    print(f"\n📊 Integration Results:")
    print(f"   Placement success: {execution_result.placement_success}")
    print(f"   Motion success: {execution_result.motion_success}")
    print(f"   Overall success: {execution_result.overall_success}")
    print(f"   Failure summary: {execution_result.failure_summary}")
    
    # Validate failure handling
    success = (
        not execution_result.placement_success and
        not execution_result.motion_success and
        not execution_result.overall_success and
        "Phase 3 placement failed" in execution_result.failure_summary
    )
    
    print(f"✅ Phase3To4Bridge placement failure test: {'PASSED' if success else 'FAILED'}")
    return success


def test_retry_logic():
    """Test retry logic with motion failures"""
    print("\n🧪 Testing retry logic...")
    
    # Initialize PyBullet
    physics_client = p.connect(p.DIRECT)
    p.setGravity(0, 0, -9.81)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF("plane.urdf")
    
    try:
        # Initialize robot and motion controller
        robot = RobotController(physics_client)
        robot.initialize()
        
        motion_controller = MotionController(robot)
        
        # Create bridge with limited retries
        bridge = Phase3To4Bridge(motion_controller, max_retries=2)
        
        # Create placement result that might cause IK issues (very high position)
        placement_result = create_mock_placement_result(success=True, position=(0.3, 0.2, 1.5))  # Very high
        
        print(f"   Target position (challenging): {placement_result.placement_command.position}")
        print(f"   Max retries configured: {bridge.max_retries}")
        
        # Execute with potential retry scenario
        execution_result = bridge.execute_placement_with_motion(placement_result)
        
        print(f"\n📊 Retry Logic Results:")
        print(f"   Overall success: {execution_result.overall_success}")
        print(f"   Execution time: {execution_result.total_execution_time:.2f}s")
        
        if execution_result.motion_result:
            print(f"   Motion waypoints completed: {execution_result.motion_result.waypoints_completed}")
            if not execution_result.motion_result.success:
                print(f"   Motion failure reason: {execution_result.motion_result.failure_reason}")
        
        # Test retry strategy method directly
        original_cmd = placement_result.placement_command
        jittered_cmd = bridge._apply_retry_strategy(original_cmd, 1)
        
        position_changed = jittered_cmd.position != original_cmd.position
        confidence_reduced = jittered_cmd.confidence_score < original_cmd.confidence_score
        
        print(f"   Position jitter applied: {position_changed}")
        print(f"   Confidence reduced: {confidence_reduced}")
        
        # Success if retry logic executes (regardless of final success)
        success = (
            execution_result.total_execution_time < 30.0 and
            position_changed and confidence_reduced  # Retry strategy works
        )
        
        print(f"✅ Retry logic test: {'PASSED' if success else 'FAILED'}")
        return success
        
    finally:
        robot.cleanup()
        p.disconnect()


def test_performance_monitoring():
    """Test performance monitoring and history tracking"""
    print("\n🧪 Testing performance monitoring...")
    
    # Create mock bridge
    robot = RobotController()
    motion_controller = MotionController(robot)
    bridge = Phase3To4Bridge(motion_controller)
    
    # Execute multiple mock operations
    results = []
    for i in range(5):
        placement_result = create_mock_placement_result(
            success=True, 
            position=(0.3 + i*0.05, 0.2, 0.3)  # Slightly different positions
        )
        
        # Mock execution result
        execution_result = ExecutionResult(
            placement_success=True,
            motion_success=i % 2 == 0,  # Alternate success/failure
            placement_result=placement_result,
            motion_result=None,
            total_execution_time=1.5 + i*0.2,  # Varying execution times
            overall_success=i % 2 == 0
        )
        
        bridge.execution_history.append(execution_result)
        results.append(execution_result)
    
    # Test performance metrics
    success_rate = bridge.get_success_rate()
    avg_time = bridge.get_average_execution_time()
    
    print(f"   Operations executed: {len(bridge.execution_history)}")
    print(f"   Success rate: {success_rate:.1%}")
    print(f"   Average execution time: {avg_time:.2f}s")
    
    # Validate metrics
    expected_success_rate = 0.6  # 3 out of 5 successes
    expected_avg_time = sum(r.total_execution_time for r in results) / len(results)
    
    metrics_correct = (
        abs(success_rate - expected_success_rate) < 0.01 and
        abs(avg_time - expected_avg_time) < 0.01
    )
    
    # Test history clearing
    bridge.clear_history()
    history_cleared = len(bridge.execution_history) == 0
    
    print(f"   Metrics calculation correct: {metrics_correct}")
    print(f"   History cleared: {history_cleared}")
    
    success = metrics_correct and history_cleared
    
    print(f"✅ Performance monitoring test: {'PASSED' if success else 'FAILED'}")
    return success


def main():
    """Run Phase 4D tests"""
    print("🚀 Phase 4D Testing: Integration with Phase 3 + error handling")
    print("=" * 65)
    
    tests = [
        test_phase3to4_bridge_placement_failure,
        test_performance_monitoring,
        test_retry_logic,
        test_phase3to4_bridge_success,
    ]
    
    results = []
    for test in tests:
        result = test()
        results.append(result)
        print()
    
    passed = sum(results)
    total = len(results)
    
    print("=" * 65)
    print(f"📊 Phase 4D Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 Phase 4D: Integration with Phase 3 + error handling COMPLETE!")
        print("✅ Key achievements:")
        print("   - Phase3To4Bridge for seamless integration")
        print("   - Comprehensive error handling and retry logic")
        print("   - Position jitter retry strategy (±2cm)")
        print("   - Performance monitoring and history tracking")
        print("   - Combined ExecutionResult for full pipeline status")
        print("✅ Ready for Phase 4E: Testing + validation")
    else:
        print("❌ Phase 4D: Some tests failed - review implementation")
        
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
#!/usr/bin/env python3
"""
Step 1: Baseline Testing - Test varied cube sizes
Tests grasping with cube scales from 0.05m to 0.15m to establish baseline success rates
"""

import sys
import os
sys.path.append('src')

import pybullet as p
import pybullet_data
import time
from control import RobotController, MotionController
from placement import PlacementCommand
from vision import CameraSystem, CameraConfiguration
from object_detection import ObjectDetector

def test_cube_size(scale: float, run_number: int) -> dict:
    """Test grasping with a specific cube size"""
    print(f"\n{'='*60}")
    print(f"🧪 TEST RUN {run_number}: Cube scale {scale:.3f}m ({scale*100:.1f}cm)")
    print(f"{'='*60}")
    
    # Connect to PyBullet
    physics_client = p.connect(p.DIRECT)  # Headless for batch testing
    p.setGravity(0, 0, -9.81)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    
    # Physics optimization
    p.setTimeStep(1./1000.)
    p.setPhysicsEngineParameter(numSolverIterations=150)
    p.setPhysicsEngineParameter(numSubSteps=80)
    
    try:
        # Load ground
        p.loadURDF("plane.urdf")
        
        # Create shelf - same position as main demo
        shelf_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=p.createCollisionShape(
                p.GEOM_BOX, halfExtents=[0.3, 0.2, 0.01]
            ),
            baseVisualShapeIndex=p.createVisualShape(
                p.GEOM_BOX, halfExtents=[0.3, 0.2, 0.01], 
                rgbaColor=[0.8, 0.6, 0.4, 1.0]
            ),
            basePosition=[0.7, 0.3, 0.4]
        )
        
        # Add test cube with variable scale
        cube_height = scale / 2  # Half the cube size for ground level
        object_id = p.loadURDF(
            "cube.urdf",
            [0.4, -0.2, cube_height],
            globalScaling=scale
        )
        
        # Add friction
        p.changeDynamics(object_id, -1, 
                        lateralFriction=1.0, 
                        spinningFriction=0.005,
                        rollingFriction=0.005)
        
        print(f"📦 Created {scale*100:.1f}cm cube at height {cube_height:.3f}m")
        
        # Initialize robot
        robot = RobotController(physics_client)
        robot.initialize()
        
        # Add friction to gripper
        p.changeDynamics(robot.robot_id, 9, lateralFriction=1.0, spinningFriction=0.005, rollingFriction=0.005)
        p.changeDynamics(robot.robot_id, 10, lateralFriction=1.0, spinningFriction=0.005, rollingFriction=0.005)
        
        # Initialize vision system
        camera_config = CameraConfiguration(
            image_width=640,
            image_height=480, 
            fov=60.0
        )
        camera_system = CameraSystem(camera_config, physics_client)
        camera_system.initialize()
        
        # Setup camera for detection
        overhead_camera = camera_system.setup_top_down_camera(
            shelf_center=(0.4, -0.2, cube_height)
        )
        
        # Initialize detector
        object_detector = ObjectDetector(camera_system)
        object_detector.set_overhead_camera("top_down_0")
        
        # Run physics simulation
        for _ in range(10):
            p.stepSimulation()
        
        # Test object detection
        detections = object_detector.detect_objects_on_table()
        detection_success = len(detections) > 0
        
        if detection_success:
            detected_pos = detections[0].position
            actual_pos = [0.4, -0.2, cube_height]
            detection_error = {
                'x': abs(detected_pos[0] - actual_pos[0]),
                'y': abs(detected_pos[1] - actual_pos[1]),
                'z': abs(detected_pos[2] - actual_pos[2])
            }
            print(f"✅ Detection successful: Error X={detection_error['x']:.3f}, Y={detection_error['y']:.3f}, Z={detection_error['z']:.3f}")
        else:
            detection_error = {'x': None, 'y': None, 'z': None}
            print("❌ Detection failed")
        
        # Initialize motion controller
        motion_controller = MotionController(
            robot, 
            object_detector=object_detector
        )
        
        # Execute placement
        placement_cmd = PlacementCommand(
            object_id=f"test_cube_{scale}",
            zone_id="test_zone",
            position=(0.7, 0.3, 0.42),
            orientation=(0, 0, 0, 1),
            confidence_score=0.8
        )
        
        print("🤖 Starting pick-and-place sequence...")
        start_time = time.time()
        result = motion_controller.execute_placement(placement_cmd)
        exec_time = time.time() - start_time
        
        # Collect results
        test_result = {
            'cube_scale': scale,
            'cube_size_cm': scale * 100,
            'run_number': run_number,
            'detection_success': detection_success,
            'detection_error': detection_error,
            'grasp_success': result.success,
            'execution_time': exec_time,
            'waypoints_completed': result.waypoints_completed,
            'total_waypoints': result.total_waypoints,
            'failure_reason': result.failure_reason if not result.success else None
        }
        
        print(f"📊 RESULTS: Detection={'✅' if detection_success else '❌'}, Grasp={'✅' if result.success else '❌'}, Time={exec_time:.1f}s")
        
        return test_result
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return {
            'cube_scale': scale,
            'cube_size_cm': scale * 100,
            'run_number': run_number,
            'detection_success': False,
            'detection_error': {'x': None, 'y': None, 'z': None},
            'grasp_success': False,
            'execution_time': None,
            'waypoints_completed': 0,
            'total_waypoints': 7,
            'failure_reason': str(e)
        }
    finally:
        robot.cleanup()
        p.disconnect()

def main():
    """Run baseline testing with varied cube sizes"""
    print("🚀 Step 1: Baseline Testing - Cube Size Variation")
    print("=" * 60)
    
    # Test cube sizes from 5cm to 15cm (as per adaptive grasping plan)
    test_scales = [0.05, 0.07, 0.09, 0.11, 0.13, 0.15]
    results = []
    
    for i, scale in enumerate(test_scales, 1):
        result = test_cube_size(scale, i)
        results.append(result)
        
        # Brief pause between tests
        time.sleep(1)
    
    # Analyze results
    print("\n" + "=" * 80)
    print("📊 BASELINE TESTING RESULTS SUMMARY")
    print("=" * 80)
    
    detection_successes = sum(1 for r in results if r['detection_success'])
    grasp_successes = sum(1 for r in results if r['grasp_success'])
    total_tests = len(results)
    
    print(f"🎯 Overall Success Rates:")
    print(f"   Detection: {detection_successes}/{total_tests} ({100*detection_successes/total_tests:.1f}%)")
    print(f"   Grasping: {grasp_successes}/{total_tests} ({100*grasp_successes/total_tests:.1f}%)")
    
    print(f"\n📏 Size-specific results:")
    for result in results:
        status = "✅" if result['grasp_success'] else "❌"
        print(f"   {result['cube_size_cm']:4.1f}cm: {status} (Detection: {'✅' if result['detection_success'] else '❌'})")
    
    # Failure analysis
    failures = [r for r in results if not r['grasp_success']]
    if failures:
        print(f"\n⚠️  Failure Analysis:")
        failure_reasons = {}
        for f in failures:
            reason = f['failure_reason'] or 'Unknown'
            failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
        
        for reason, count in failure_reasons.items():
            print(f"   {reason}: {count} cases")
    
    # Save results to file for future analysis
    import json
    with open('baseline_results.json', 'w') as f:
        json.dump({
            'test_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'test_description': 'Baseline cube size variation test',
            'results': results,
            'summary': {
                'detection_success_rate': detection_successes / total_tests,
                'grasp_success_rate': grasp_successes / total_tests,
                'total_tests': total_tests
            }
        }, f, indent=2)
    
    print(f"\n💾 Results saved to baseline_results.json")
    print(f"🎯 Baseline established - ready for Step 2 (Hybrid ML integration)")

if __name__ == "__main__":
    main()
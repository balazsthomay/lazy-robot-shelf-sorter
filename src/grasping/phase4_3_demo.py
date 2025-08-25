"""
Phase 4.3 Demo - Adaptive Behavior + Enhanced Feedback

Demonstrates the complete Phase 4.3 system with:
- Enhanced GraspExecutor with detailed metrics
- Real-time performance tracking
- Success rate measurement
- Quantified feedback system
"""

import pybullet as p
import numpy as np
import time
import cv2
import os
from pathlib import Path

import sys
from pathlib import Path

try:
    # Try relative imports (when run as module)
    from .predictor import GGCNNPredictor
    from .coordinate_transforms import VisionSystemIntegrator
    from .planner import MotionPlanner
    from .executor import GraspExecutor
    from .performance_tracker import PerformanceTracker, get_performance_tracker
except ImportError:
    # Add project root to path when running directly
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))
    from src.grasping.predictor import GGCNNPredictor
    from src.grasping.coordinate_transforms import VisionSystemIntegrator
    from src.grasping.planner import MotionPlanner
    from src.grasping.executor import GraspExecutor
    from src.grasping.performance_tracker import PerformanceTracker, get_performance_tracker


def setup_simulation():
    """Setup PyBullet simulation environment."""
    physics_client = p.connect(p.GUI)
    p.resetDebugVisualizerCamera(cameraDistance=1.8, cameraYaw=45, 
                               cameraPitch=-30, cameraTargetPosition=[0.0, 0.2, 0.6])
    
    # Add PyBullet data path
    import pybullet_data
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    
    # Load ground plane
    plane_id = p.loadURDF("plane.urdf")
    
    # Load Franka Panda robot beside the table
    robot_id = p.loadURDF(
        "franka_panda/panda.urdf",
        basePosition=[0.5, 0.2, 0.0],
        useFixedBase=True
    )
    
    # Set robot to starting position
    joint_positions = [0, -0.785, 0, -2.356, 0, 1.571, 0.785, 0.04, 0.04]
    for i, pos in enumerate(joint_positions):
        p.resetJointState(robot_id, i, pos)
    
    # Create table surface
    table_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.4, 0.4, 0.03])
    table_body = p.createMultiBody(0, table_id, -1, [0.0, 0.2, 0.60], [0, 0, 0, 1])
    p.changeVisualShape(table_body, -1, rgbaColor=[0.8, 0.8, 0.6, 1.0])
    
    # Set gravity
    p.setGravity(0, 0, -9.81)
    
    return physics_client, robot_id


def setup_camera():
    """Setup camera for RGB-D capture."""
    camera_pos = [0.2, -0.3, 1.2]
    camera_target = [0.0, 0.2, 0.68]
    camera_up = [0, 0, 1]
    
    return {
        'position': camera_pos,
        'target': camera_target,
        'up': camera_up,
        'width': 640,
        'height': 480,
        'fov': 60,
        'aspect': 640 / 480,
        'near': 0.1,
        'far': 10.0
    }


def create_test_objects():
    """Create various test objects for demonstration."""
    objects = []
    
    # Different object types to test diversity
    object_configs = [
        {
            'name': 'cylinder_can',
            'category': 'Can',
            'shape': p.GEOM_CYLINDER,
            'params': {'radius': 0.032, 'height': 0.10},
            'mass': 0.3,
            'color': [0.8, 0.2, 0.2, 1.0],
            'position': [0.0, 0.2, 0.68]
        },
        {
            'name': 'small_box',
            'category': 'Box', 
            'shape': p.GEOM_BOX,
            'params': {'halfExtents': [0.03, 0.05, 0.06]},
            'mass': 0.25,
            'color': [0.2, 0.8, 0.2, 1.0],
            'position': [0.08, 0.15, 0.68]
        },
        {
            'name': 'bottle_shape',
            'category': 'Bottle',
            'shape': p.GEOM_CYLINDER,
            'params': {'radius': 0.025, 'height': 0.12},
            'mass': 0.2,
            'color': [0.2, 0.2, 0.8, 1.0],
            'position': [-0.08, 0.25, 0.68]
        }
    ]
    
    return object_configs


def load_single_object(obj_config):
    """Load a single object based on configuration."""
    if obj_config['shape'] == p.GEOM_CYLINDER:
        collision_shape = p.createCollisionShape(
            obj_config['shape'], 
            radius=obj_config['params']['radius'],
            height=obj_config['params']['height']
        )
    elif obj_config['shape'] == p.GEOM_BOX:
        collision_shape = p.createCollisionShape(
            obj_config['shape'],
            halfExtents=obj_config['params']['halfExtents']
        )
    else:
        collision_shape = p.createCollisionShape(
            p.GEOM_SPHERE,
            radius=0.03
        )
    
    body_id = p.createMultiBody(
        obj_config['mass'],
        collision_shape,
        -1,
        obj_config['position'],
        [0, 0, 0, 1]
    )
    
    p.changeVisualShape(body_id, -1, rgbaColor=obj_config['color'])
    
    return body_id


def capture_rgbd(camera_params):
    """Capture RGB-D images from camera."""
    view_matrix = p.computeViewMatrix(
        cameraEyePosition=camera_params['position'],
        cameraTargetPosition=camera_params['target'],
        cameraUpVector=camera_params['up']
    )
    
    proj_matrix = p.computeProjectionMatrixFOV(
        fov=camera_params['fov'],
        aspect=camera_params['aspect'],
        nearVal=camera_params['near'],
        farVal=camera_params['far']
    )
    
    width, height = camera_params['width'], camera_params['height']
    
    _, _, rgb_img, depth_img, _ = p.getCameraImage(
        width=width,
        height=height,
        viewMatrix=view_matrix,
        projectionMatrix=proj_matrix,
        renderer=p.ER_BULLET_HARDWARE_OPENGL
    )
    
    # Convert to numpy arrays
    rgb_array = np.array(rgb_img).reshape(height, width, 4)[:, :, :3]
    depth_array = np.array(depth_img).reshape(height, width)
    
    # Convert depth to meters
    near, far = camera_params['near'], camera_params['far']
    depth_meters = far * near / (far - (far - near) * depth_array)
    
    return rgb_array, depth_meters, view_matrix, proj_matrix


def visualize_enhanced_metrics(rgb_img, result, object_name, save_path):
    """Visualize enhanced execution metrics."""
    vis_img = rgb_img.copy()
    
    # Enhanced metrics display
    y_offset = 30
    line_height = 25
    
    # Main success indicator
    success_color = (0, 255, 0) if result.success else (0, 0, 255)
    status_text = f"RESULT: {'SUCCESS' if result.success else 'FAILED'}"
    cv2.putText(vis_img, status_text, (10, y_offset), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, success_color, 2)
    y_offset += line_height + 5
    
    # Object info
    cv2.putText(vis_img, f"Object: {object_name}", (10, y_offset),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    y_offset += line_height
    
    # Core metrics
    cv2.putText(vis_img, f"Force: {result.gripper_force:.1f}N", (10, y_offset),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    y_offset += line_height
    
    cv2.putText(vis_img, f"Lift: {result.lift_height_achieved:.3f}m", (10, y_offset),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    y_offset += line_height
    
    cv2.putText(vis_img, f"Object Moved: {result.object_moved}", (10, y_offset),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    y_offset += line_height
    
    # Enhanced metrics (if available)
    if result.metrics:
        cv2.putText(vis_img, f"Stability: {result.metrics.grasp_stability_score:.2f}", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += line_height
        
        cv2.putText(vis_img, f"Contacts: {result.metrics.contact_points}", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += line_height
        
        cv2.putText(vis_img, f"Exec Time: {result.metrics.execution_time:.1f}s", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += line_height
        
        cv2.putText(vis_img, f"Force Consistency: {result.metrics.force_consistency:.2f}", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Retry info
    if result.retry_count > 0:
        cv2.putText(vis_img, f"Retries: {result.retry_count}", (10, y_offset + line_height),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    # Save visualization
    os.makedirs("debug_output", exist_ok=True)
    cv2.imwrite(save_path, cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))
    print(f"📸 Enhanced visualization saved to {save_path}")


def demo_single_object(predictor, grasp_executor, camera_params, 
                      tracker, obj_config, demo_number):
    """Demonstrate grasping with enhanced metrics on single object."""
    print(f"\\n{'='*60}")
    print(f"🎯 PHASE 4.3 DEMO {demo_number}: {obj_config['name'].upper()}")
    print(f"   Category: {obj_config['category']} | Expected: Medium challenge")
    print(f"{'='*60}")
    
    # Load object
    print("\\n1. Loading test object...")
    obj_id = load_single_object(obj_config)
    
    # Let simulation settle
    for _ in range(100):
        p.stepSimulation()
    
    print("\\n2. Capturing scene and predicting grasps...")
    # Capture RGB-D
    rgb_img, depth_img, _, _ = capture_rgbd(camera_params)
    print(f"   ✓ Captured images: RGB {rgb_img.shape}, depth range {depth_img.min():.3f}-{depth_img.max():.3f}m")
    
    # Get grasp predictions
    grasp_poses = predictor.predict(rgb_img, depth_img)
    print(f"   ✓ Generated {len(grasp_poses)} grasp candidates")
    
    if not grasp_poses:
        print("   ⚠️ No grasp predictions - skipping execution")
        p.removeBody(obj_id)
        return
    
    # Show top predictions
    print("\\n3. Top grasp candidates:")
    for i, grasp in enumerate(grasp_poses[:3]):
        print(f"   Grasp {i+1}: conf={grasp.confidence:.3f}, "
              f"pos=({grasp.position[0]:.3f}, {grasp.position[1]:.3f}, {grasp.position[2]:.3f})")
    
    print("\\n4. Executing grasp with enhanced feedback...")
    print("   🤖 Watch the PyBullet GUI - robot will move with smooth motion")
    print("   📊 Enhanced metrics will be measured during execution")
    
    # Execute grasp with retry
    start_time = time.time()
    result = grasp_executor.execute_with_retry(grasp_poses)
    end_time = time.time()
    
    # Record performance for tracking
    best_confidence = grasp_poses[0].confidence if grasp_poses else 0.0
    tracker.record_attempt(result, obj_config['name'], obj_config['category'], best_confidence)
    
    print("\\n5. 📈 ENHANCED EXECUTION RESULTS:")
    print(f"   {'🎯 SUCCESS' if result.success else '❌ FAILED'}: Grasp execution")
    
    if result.metrics:
        print(f"   📊 Detailed Metrics:")
        print(f"      Max Force: {result.metrics.max_gripper_force:.1f}N")
        print(f"      Avg Force: {result.metrics.avg_gripper_force:.1f}N") 
        print(f"      Force Consistency: {result.metrics.force_consistency:.2f}")
        print(f"      Contact Points: {result.metrics.contact_points}")
        print(f"      Stability Score: {result.metrics.grasp_stability_score:.2f}")
        print(f"      Object Displacement: {result.metrics.object_displacement:.4f}m")
        print(f"      Execution Time: {result.metrics.execution_time:.1f}s")
    
    print(f"   🔄 Retry Count: {result.retry_count}")
    if result.error_message:
        print(f"   ⚠️ Error: {result.error_message}")
    
    # Save visualization with enhanced metrics
    vis_filename = f"debug_output/phase4_3_demo_{demo_number}_{obj_config['name']}.png"
    visualize_enhanced_metrics(rgb_img, result, obj_config['name'], vis_filename)
    
    # Clean up
    p.removeBody(obj_id)
    
    print(f"\\n6. ✅ Demo {demo_number} complete - object removed")
    return result


def main():
    """Main Phase 4.3 demo function."""
    print("🚀 PHASE 4.3 DEMO: ADAPTIVE BEHAVIOR + ENHANCED FEEDBACK")
    print("=" * 70)
    print("Features demonstrated:")
    print("  • Enhanced GraspExecutor with detailed force/stability metrics") 
    print("  • Real-time performance tracking and analytics")
    print("  • Success rate measurement with trend analysis")
    print("  • Multi-criteria success detection (5 factors)")
    print("  • Quantified feedback system with comprehensive logging")
    print("=" * 70)
    
    # Setup simulation
    physics_client, robot_id = setup_simulation()
    camera_params = setup_camera()
    
    # Initialize performance tracker
    tracker = get_performance_tracker()
    
    try:
        # Initialize all components
        predictor = GGCNNPredictor(device='cpu', max_predictions=3)
        
        # Load model
        model_path = Path(__file__).parent.parent.parent / "data" / "models" / "ggcnn_epoch_23_cornell_statedict.pt"
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at {model_path}")
        predictor.load_model(str(model_path))
        print("✓ GG-CNN predictor loaded")
        
        # Setup coordinate transformer
        vision_integrator = VisionSystemIntegrator()
        transformer = vision_integrator.setup_from_camera_system(
            camera_pos=np.array(camera_params['position']),
            camera_target=np.array(camera_params['target']),
            camera_up=np.array(camera_params['up']),
            fov=camera_params['fov'],
            aspect=camera_params['aspect'],
            near=camera_params['near'],
            far=camera_params['far'],
            width=camera_params['width'],
            height=camera_params['height']
        )
        predictor.set_coordinate_transformer(transformer)
        print("✓ Coordinate transformer configured")
        
        # Initialize motion planner and executor with enhanced feedback
        motion_planner = MotionPlanner(robot_id)
        grasp_executor = GraspExecutor(robot_id, motion_planner)
        print("✓ Enhanced motion planner and executor initialized")
        
        # Get test objects
        test_objects = create_test_objects()
        
        # Let simulation settle
        for _ in range(100):
            p.stepSimulation()
        
        print(f"\\n🎯 Running Phase 4.3 demo on {len(test_objects)} diverse objects...")
        
        # Demo each object type
        results = []
        for i, obj_config in enumerate(test_objects):
            result = demo_single_object(
                predictor, grasp_executor, camera_params, 
                tracker, obj_config, i + 1
            )
            if result:
                results.append(result)
            
            # Show real-time dashboard after each attempt
            tracker.print_realtime_dashboard()
            
            if i < len(test_objects) - 1:
                print("\\n⏸️ Press Enter to continue to next object demo...")
                try:
                    input()
                except (EOFError, KeyboardInterrupt):
                    print("Continuing automatically...")
                    time.sleep(2)
        
        # Final performance summary
        print("\\n" + "="*70)
        print("🏁 PHASE 4.3 DEMO COMPLETE - FINAL PERFORMANCE SUMMARY")
        print("="*70)
        
        # Get final statistics
        session_stats = tracker.get_current_session_stats()
        trends = tracker.analyze_performance_trends()
        
        print(f"\\n📊 SESSION RESULTS:")
        print(f"   Total Attempts: {session_stats['total_attempts']}")
        print(f"   Success Rate: {session_stats['success_rate']:.1%}")
        print(f"   Performance Rating: {session_stats['performance_rating'].upper()}")
        print(f"   Avg Execution Time: {session_stats['avg_execution_time']:.1f}s")
        print(f"   Avg Force: {session_stats['avg_force']:.1f}N")
        print(f"   Avg Stability: {session_stats['avg_stability']:.2f}")
        
        # Phase 4.3 success gate evaluation
        success_gate_met = session_stats['success_rate'] >= 0.5
        print(f"\\n🎯 PHASE 4.3 SUCCESS GATE (≥50%):")
        if success_gate_met:
            print(f"   ✅ PASSED: {session_stats['success_rate']:.1%} success rate")
            print(f"   🚀 Ready to proceed to Phase 4.4!")
        else:
            print(f"   ❌ FAILED: {session_stats['success_rate']:.1%} success rate")
            print(f"   🔥 Data generation pipeline would be activated")
            print(f"   📈 Need to generate training data and fine-tune GG-CNN")
        
        print(f"\\n📈 PERFORMANCE TRENDS:")
        print(f"   Overall Trend: {trends.overall_trend.upper()}")
        if trends.improvement_rate != 0:
            print(f"   Improvement Rate: {trends.improvement_rate:+.1f}% per hour")
        
        # Category analysis
        if session_stats['category_breakdown']:
            print(f"\\n📦 CATEGORY PERFORMANCE:")
            for category, rate in sorted(session_stats['category_breakdown'].items(),
                                       key=lambda x: x[1], reverse=True):
                emoji = "🟢" if rate >= 0.6 else "🟡" if rate >= 0.3 else "🔴"
                print(f"   {emoji} {category}: {rate:.1%}")
        
        # Save performance log
        log_file = tracker.save_performance_log()
        print(f"\\n💾 Complete performance log saved to: {log_file}")
        
        print(f"\\n🎉 Phase 4.3 demonstration complete!")
        print("   Enhanced feedback system provides detailed metrics for each grasp")
        print("   Performance tracking enables data-driven improvements")
        print("   Ready for baseline testing or Phase 4.4 integration")
        
        # Wait for user observation
        print("\\n👀 Observe the PyBullet GUI and check debug_output/ for visualizations")
        print("Press Enter to exit...")
        
        try:
            input()
        except (EOFError, KeyboardInterrupt):
            print("Demo completed.")
        
    except Exception as e:
        print(f"✗ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Cleanup
        p.disconnect(physics_client)
        
    print("\\n=== Phase 4.3 Demo Complete ===")


if __name__ == "__main__":
    main()
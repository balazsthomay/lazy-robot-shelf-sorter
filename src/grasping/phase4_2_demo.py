"""
Phase 4.2 Demo - Enhanced Control with ML Guidance

Demonstrates complete grasping pipeline with motion planning and execution:
- GG-CNN grasp prediction
- Motion planning with orientation constraints
- Grasp execution with feedback
- Success detection and retry logic
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
except ImportError:
    # Add project root to path when running directly
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))
    from src.grasping.predictor import GGCNNPredictor
    from src.grasping.coordinate_transforms import VisionSystemIntegrator
    from src.grasping.planner import MotionPlanner
    from src.grasping.executor import GraspExecutor


def setup_simulation():
    """Setup PyBullet simulation environment with better object positioning."""
    # Use GUI mode to see execution
    physics_client = p.connect(p.GUI)
    p.resetDebugVisualizerCamera(cameraDistance=1.8, cameraYaw=45, cameraPitch=-30, cameraTargetPosition=[0.0, 0.2, 0.6])
    
    # Add PyBullet data path
    import pybullet_data
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    
    # Load ground plane
    plane_id = p.loadURDF("plane.urdf")
    
    # Load Franka Panda robot beside the table, not underneath it
    robot_id = p.loadURDF(
        "franka_panda/panda.urdf",
        basePosition=[0.5, 0.2, 0.0],  # Beside table at ground level
        useFixedBase=True
    )
    
    # Set robot to better starting position
    joint_positions = [0, -0.785, 0, -2.356, 0, 1.571, 0.785, 0.04, 0.04]
    for i, pos in enumerate(joint_positions):
        p.resetJointState(robot_id, i, pos)
    
    # Load test objects in better positions for grasping
    objects = []
    
    # Create table surface first - stable base
    table_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.4, 0.4, 0.03])
    table_body = p.createMultiBody(0, table_id, -1, [0.0, 0.2, 0.60], [0, 0, 0, 1])
    p.changeVisualShape(table_body, -1, rgbaColor=[0.8, 0.8, 0.6, 1.0])
    
    # Single object for focused testing - positioned stably on table
    pos = [0.0, 0.2, 0.68]  # On top of table surface (0.60 + 0.03 + 0.05 for clearance)
    
    # Create a graspable cylinder - fixed in place for stable testing
    obj_id = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.025, height=0.08)
    obj_body = p.createMultiBody(0.05, obj_id, -1, pos, [0, 0, 0, 1])  # Create object
    # Make it fixed in place by setting mass to 0 after creation
    p.changeDynamics(obj_body, -1, mass=0)
    p.changeVisualShape(obj_body, -1, rgbaColor=[0.8, 0.2, 0.2, 1.0])
    
    objects.append(obj_body)
    
    # Set gravity
    p.setGravity(0, 0, -9.81)
    
    return physics_client, robot_id, objects


def setup_camera():
    """Setup camera for RGB-D capture positioned for workspace view."""
    # Camera positioned slightly angled above object for optimal view
    camera_pos = [0.2, -0.3, 1.2]  # Slightly offset from directly above
    camera_target = [0.0, 0.2, 0.68]  # Look at object on table
    camera_up = [0, 0, 1]
    
    # Camera parameters
    width, height = 640, 480
    fov = 60
    aspect = width / height
    near, far = 0.1, 10.0
    
    return {
        'position': camera_pos,
        'target': camera_target,
        'up': camera_up,
        'width': width,
        'height': height,
        'fov': fov,
        'aspect': aspect,
        'near': near,
        'far': far
    }


def capture_rgbd(camera_params):
    """Capture RGB-D images from camera."""
    # Compute view and projection matrices
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
    
    # Capture images
    width, height = camera_params['width'], camera_params['height']
    
    _, _, rgb_img, depth_img, _ = p.getCameraImage(
        width=width,
        height=height,
        viewMatrix=view_matrix,
        projectionMatrix=proj_matrix,
        renderer=p.ER_BULLET_HARDWARE_OPENGL
    )
    
    # Convert to numpy arrays
    rgb_array = np.array(rgb_img).reshape(height, width, 4)[:, :, :3]  # Remove alpha
    depth_array = np.array(depth_img).reshape(height, width)
    
    # Convert depth from normalized to meters
    near, far = camera_params['near'], camera_params['far']
    depth_meters = far * near / (far - (far - near) * depth_array)
    
    return rgb_array, depth_meters, view_matrix, proj_matrix


def visualize_execution_state(rgb_img, grasp_poses, current_grasp_idx, execution_result, save_path):
    """Visualize the current execution state."""
    vis_img = rgb_img.copy()
    
    # Draw all grasp predictions
    for i, grasp in enumerate(grasp_poses):
        color = (0, 255, 0) if i == current_grasp_idx else (128, 128, 128)
        text = f"Grasp {i+1}: conf={grasp.confidence:.2f}"
        cv2.putText(vis_img, text, (10, 30 + i*25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    # Draw execution status
    if execution_result:
        status_text = f"Execution: {'SUCCESS' if execution_result.success else 'FAILED'}"
        status_color = (0, 255, 0) if execution_result.success else (0, 0, 255)
        cv2.putText(vis_img, status_text, (10, 200), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        cv2.putText(vis_img, f"Force: {execution_result.gripper_force:.1f}N", (10, 230), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(vis_img, f"Moved: {execution_result.object_moved}", (10, 250), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(vis_img, f"Lift: {execution_result.lift_height_achieved:.3f}m", (10, 270), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Save visualization
    os.makedirs("debug_output", exist_ok=True)
    cv2.imwrite(save_path, cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))
    print(f"Execution visualization saved to {save_path}")


def main():
    """Main Phase 4.2 demo function."""
    print("=== Phase 4.2 Demo: Enhanced Control with ML Guidance ===")
    
    # Setup simulation
    physics_client, robot_id, objects = setup_simulation()
    camera_params = setup_camera()
    
    try:
        # Initialize all components
        predictor = GGCNNPredictor(device='cpu', max_predictions=3)  # Fewer candidates for focused testing
        
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
        
        # Initialize motion planner and executor
        motion_planner = MotionPlanner(robot_id)
        grasp_executor = GraspExecutor(robot_id, motion_planner)
        print("✓ Motion planner and executor initialized")
        
        # Let simulation settle
        for _ in range(100):
            p.stepSimulation()
        
        print("\nPhase 4.2 Pipeline Execution:")
        print("1. Capturing scene and predicting grasps...")
        
        # Capture RGB-D images
        rgb_img, depth_img, _, _ = capture_rgbd(camera_params)
        print(f"✓ Captured images: RGB {rgb_img.shape}, depth range {depth_img.min():.3f}-{depth_img.max():.3f}m")
        
        # Run GG-CNN prediction
        grasp_poses = predictor.predict(rgb_img, depth_img)
        print(f"✓ Generated {len(grasp_poses)} grasp candidates")
        
        # Debug: Check object position in world coordinates
        if objects:
            obj_pos, _ = p.getBasePositionAndOrientation(objects[0])
            print(f"   Debug: Actual object position: ({obj_pos[0]:.3f}, {obj_pos[1]:.3f}, {obj_pos[2]:.3f})")
        
        if not grasp_poses:
            print("⚠ No grasp predictions generated")
            return
        
        # Display grasp candidates
        print("\n2. Evaluating grasp candidates:")
        reachable_grasps = []
        for i, grasp in enumerate(grasp_poses):
            reachable = motion_planner.validate_reachability(grasp)
            reachable_grasps.append(reachable)
            status = "REACHABLE" if reachable else "UNREACHABLE"
            print(f"   Grasp {i+1}: conf={grasp.confidence:.3f}, pos=({grasp.position[0]:.3f}, {grasp.position[1]:.3f}, {grasp.position[2]:.3f}), {status}")
        
        # Find first reachable grasp
        reachable_indices = [i for i, reachable in enumerate(reachable_grasps) if reachable]
        
        if not reachable_indices:
            print("⚠ No reachable grasps found")
            print("   🧪 TESTING: Forcing execution with best grasp to demonstrate slow motion")
            # Force execution with the highest confidence grasp for testing
            reachable_indices = [0]  # Use first (highest confidence) grasp
            
        if not reachable_indices:
            visualize_execution_state(rgb_img, grasp_poses, -1, None, "debug_output/phase4_2_no_reachable.png")
            return
        
        # Execute best reachable grasp
        best_idx = reachable_indices[0]
        best_grasp = grasp_poses[best_idx]
        
        print(f"\n3. Executing grasp {best_idx + 1} (confidence: {best_grasp.confidence:.3f})...")
        print("   Watch the PyBullet GUI window - robot will move slowly for observation")
        
        # Execute with retry using all reachable candidates
        reachable_grasp_poses = [grasp_poses[i] for i in reachable_indices]
        execution_result = grasp_executor.execute_with_retry(reachable_grasp_poses)
        
        print(f"\n4. Execution Results:")
        print(f"   Success: {execution_result.success}")
        print(f"   Gripper force: {execution_result.gripper_force:.1f}N")
        print(f"   Object moved: {execution_result.object_moved}")
        print(f"   Lift achieved: {execution_result.lift_height_achieved:.3f}m")
        print(f"   Retry count: {execution_result.retry_count}")
        
        if execution_result.error_message:
            print(f"   Error: {execution_result.error_message}")
        
        # Visualize final state
        visualize_execution_state(rgb_img, grasp_poses, best_idx, execution_result, 
                                "debug_output/phase4_2_execution_result.png")
        
        # Wait for user to observe result
        print(f"\n5. Pipeline Complete!")
        print(f"   Overall success: {'✓' if execution_result.success else '✗'}")
        print("   Watch the PyBullet GUI window to see the robot execution.")
        print("   Press Enter when ready to exit...")
        
        # Keep simulation running for observation
        while True:
            try:
                user_input = input()
                break
            except (EOFError, KeyboardInterrupt):
                # Handle case where input is interrupted
                print("Simulation will continue running...")
                time.sleep(2)
                break
        
    except Exception as e:
        print(f"✗ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Cleanup
        p.disconnect(physics_client)
        
    print("\n=== Phase 4.2 Demo Complete ===")


if __name__ == "__main__":
    main()
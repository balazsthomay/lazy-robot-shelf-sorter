"""
Phase 4.1 Demo - GG-CNN Foundation with Option 1 Model Loading

Demonstrates GG-CNN grasp prediction with proper coordinate transforms
using the complete pre-trained model (Option 1 loading approach).
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
except ImportError:
    # Add project root to path when running directly
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))
    from src.grasping.predictor import GGCNNPredictor
    from src.grasping.coordinate_transforms import VisionSystemIntegrator
    from src.grasping.planner import MotionPlanner


def setup_simulation():
    """Setup PyBullet simulation environment."""
    # Use headless mode to avoid GUI freezing issues
    physics_client = p.connect(p.DIRECT)
    
    # Add PyBullet data path
    import pybullet_data
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    
    # Load ground plane
    plane_id = p.loadURDF("plane.urdf")
    
    # Load Franka Panda robot
    robot_id = p.loadURDF(
        "franka_panda/panda.urdf",
        basePosition=[0, 0, 0],
        useFixedBase=True
    )
    
    # Load test objects in robot workspace
    objects = []
    
    # YCB-style objects at reachable positions
    positions = [
        [0.4, 0.0, 0.62],   # Forward center
        [0.35, 0.15, 0.62], # Forward right
        [0.35, -0.15, 0.62] # Forward left
    ]
    
    for i, pos in enumerate(positions):
        if i == 0:
            # Use cylinder as test object
            obj_id = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.03, height=0.08)
            obj_body = p.createMultiBody(0.1, obj_id, -1, pos, [0, 0, 0, 1])
            p.changeVisualShape(obj_body, -1, rgbaColor=[0.8, 0.2, 0.2, 1.0])
        elif i == 1:
            # Use box as test object
            obj_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.025, 0.025, 0.04])
            obj_body = p.createMultiBody(0.1, obj_id, -1, pos, [0, 0, 0, 1])
            p.changeVisualShape(obj_body, -1, rgbaColor=[0.2, 0.8, 0.2, 1.0])
        else:
            # Use sphere as test object
            obj_id = p.createCollisionShape(p.GEOM_SPHERE, radius=0.03)
            obj_body = p.createMultiBody(0.1, obj_id, -1, pos, [0, 0, 0, 1])
            p.changeVisualShape(obj_body, -1, rgbaColor=[0.2, 0.2, 0.8, 1.0])
            
        objects.append(obj_body)
    
    # Set gravity
    p.setGravity(0, 0, -9.81)
    
    return physics_client, robot_id, objects


def setup_camera():
    """Setup camera for RGB-D capture with proper positioning."""
    # Camera positioned to see robot workspace
    camera_pos = [0.0, -0.5, 1.0]
    camera_target = [0.4, 0.0, 0.62]
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


def visualize_grasp_predictions(rgb_img, grasp_poses, save_path="grasp_predictions.png"):
    """Visualize grasp predictions on RGB image and save to file."""
    vis_img = rgb_img.copy()
    
    for i, grasp in enumerate(grasp_poses):
        # For visualization, we'd need to project world coordinates back to pixels
        # For now, just add text overlay
        text = f"Grasp {i+1}: conf={grasp.confidence:.2f}"
        cv2.putText(vis_img, text, (10, 30 + i*30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Save visualization
    cv2.imwrite(save_path, cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))
    print(f"Grasp visualization saved to {save_path}")
    
    return vis_img


def main():
    """Main Phase 4.1 demo function."""
    print("=== Phase 4.1 Demo: GG-CNN Foundation with Option 1 Loading ===")
    
    # Setup simulation
    physics_client, robot_id, objects = setup_simulation()
    camera_params = setup_camera()
    
    try:
        # Initialize GG-CNN predictor with Option 1 loading
        predictor = GGCNNPredictor(device='cpu', max_predictions=5)
        
        # Load the state dict for our GG-CNN architecture
        model_path = Path(__file__).parent.parent.parent / "data" / "models" / "ggcnn_epoch_23_cornell_statedict.pt"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at {model_path}")
            
        predictor.load_model(str(model_path))
        print(f"✓ Loaded complete GG-CNN model from {model_path}")
        
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
        
        # Let simulation settle
        for _ in range(100):
            p.stepSimulation()
        
        # Capture RGB-D images
        rgb_img, depth_img, view_matrix, proj_matrix = capture_rgbd(camera_params)
        
        print(f"✓ Captured RGB-D images: {rgb_img.shape}, depth range: {depth_img.min():.3f}-{depth_img.max():.3f}m")
        
        # Save input images for debugging
        os.makedirs("debug_output", exist_ok=True)
        cv2.imwrite("debug_output/input_rgb.png", cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))
        
        # Normalize depth for visualization
        depth_vis = (depth_img - depth_img.min()) / (depth_img.max() - depth_img.min()) * 255
        cv2.imwrite("debug_output/input_depth.png", depth_vis.astype(np.uint8))
        
        # Run GG-CNN prediction
        print("Running GG-CNN grasp prediction...")
        grasp_poses = predictor.predict(rgb_img, depth_img)
        
        print(f"✓ Generated {len(grasp_poses)} grasp predictions")
        
        # Display results
        if grasp_poses:
            print("\nGrasp Predictions:")
            for i, grasp in enumerate(grasp_poses):
                print(f"  {i+1}. Position: ({grasp.position[0]:.3f}, {grasp.position[1]:.3f}, {grasp.position[2]:.3f})")
                print(f"      Confidence: {grasp.confidence:.3f}, Width: {grasp.width:.3f}m")
                
            # Visualize predictions
            visualize_grasp_predictions(rgb_img, grasp_poses, "debug_output/grasp_predictions.png")
            
            # Test motion planning for best grasp
            motion_planner = MotionPlanner(robot_id)
            best_grasp = grasp_poses[0]
            
            print(f"\nTesting reachability for best grasp...")
            reachable = motion_planner.validate_reachability(best_grasp)
            print(f"✓ Best grasp {'IS' if reachable else 'IS NOT'} reachable")
            
            if reachable:
                trajectory = motion_planner.plan_grasp_approach(best_grasp)
                print(f"✓ Planned trajectory with {len(trajectory.joint_positions)} waypoints")
                
        else:
            print("⚠ No grasp predictions generated")
            
    except Exception as e:
        print(f"✗ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Cleanup
        p.disconnect(physics_client)
        
    print("\n=== Phase 4.1 Demo Complete ===")
    print("Check debug_output/ directory for saved images")


if __name__ == "__main__":
    main()
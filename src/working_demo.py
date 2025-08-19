#!/usr/bin/env python3
"""
Working Visual Demo - Solves macOS GUI freezing issue
Shows Phase 1 Foundation components with proper PyBullet GUI handling
"""

import time
import pybullet as p
import pybullet_data
from simulation import ShelfEnvironment, ShelfConfiguration
from control import RobotController


def run_working_demo():
    """Working demo using proper Phase 1 components with shared physics client"""
    print("🚀 Starting Working Visual Demo...")
    print("This shows actual Phase 1 Foundation components!")
    
    # Connect to GUI
    physics_client = p.connect(p.GUI)
    
    # CRITICAL: Start simulation immediately to fix macOS GUI freezing
    p.setGravity(0, 0, -9.81)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    
    # Load ground immediately and step simulation
    ground_id = p.loadURDF("plane.urdf")
    p.stepSimulation()  # This fixes macOS GUI freezing!
    
    print("✅ GUI window should now be responsive!")
    
    # Set better camera view
    p.resetDebugVisualizerCamera(
        cameraDistance=4.0,
        cameraYaw=45,
        cameraPitch=-30,
        cameraTargetPosition=[0, 0, 0.5]
    )
    
    print("📚 Creating shelf environment using ShelfEnvironment class...")
    
    # Use actual ShelfEnvironment with existing physics client
    config = ShelfConfiguration(num_shelves=3, shelf_spacing=0.4)
    env = ShelfEnvironment(config, physics_client=physics_client)
    env.initialize(use_gui=False)  # Don't create new connection
    shelf_ids = env.create_shelves()
    shelf_positions = env.get_shelf_positions()
    
    print(f"  ✅ Created {len(shelf_ids)} static shelves (won't fall!)")
    print(f"  ✅ Got {len(shelf_positions)} shelf positions")
    
    # Step simulation to update display
    for _ in range(10):
        p.stepSimulation()
        time.sleep(0.01)
    
    print("🤖 Loading robot using RobotController class...")
    
    # Use actual RobotController with existing physics client
    robot = RobotController(physics_client=physics_client)
    robot.initialize()
    
    print(f"  ✅ Robot loaded with {robot.num_joints} joints")
    print(f"  ✅ Found {len(robot.joint_indices)} controllable joints")
    
    # Step simulation
    for _ in range(10):
        p.stepSimulation()
        time.sleep(0.01)
    
    print("📦 Placing objects on shelves...")
    
    # Place objects using actual YCB-style objects or simple shapes
    objects = []
    colors = [[1,0,0,1], [0,1,0,1], [0,0,1,1], [1,1,0,1], [1,0,1,1], [0,1,1,1]]
    object_names = ["cube", "sphere2", "cube", "sphere2", "cube", "sphere2"]
    
    for i, (x, y, z) in enumerate(shelf_positions[:6]):
        try:
            obj_name = object_names[i % len(object_names)]
            obj_id = p.loadURDF(
                f"{obj_name}.urdf", 
                [x + 0.02*i, y, z + 0.05], 
                globalScaling=0.05
            )
            
            # Color the object
            p.changeVisualShape(obj_id, -1, rgbaColor=colors[i % len(colors)])
            objects.append(obj_id)
            print(f"  ✅ Placed {obj_name} on shelf {i//2 + 1}, level {i%2 + 1}")
            
            # Step simulation after each object
            for _ in range(5):
                p.stepSimulation()
                time.sleep(0.01)
                
        except Exception as e:
            print(f"  ⚠️ Could not load object {i}: {e}")
    
    print("📸 Testing camera system...")
    
    # Test camera captures using direct PyBullet (simulating our CameraSystem)
    camera_positions = [
        ([2.0, -1.0, 1.2], [0, 0, 0.6], "Front-Side Camera"),
        ([0, 0, 2.5], [0, 0, 0.6], "Top-Down Camera"),
        ([-1.5, 1.0, 1.0], [0, 0, 0.4], "Back-Side Camera")
    ]
    
    for pos, target, name in camera_positions:
        try:
            view_matrix = p.computeViewMatrix(pos, target, [0, 0, 1])
            proj_matrix = p.computeProjectionMatrixFOV(60, 1.0, 0.1, 10.0)
            
            images = p.getCameraImage(
                320, 240,  # Larger size for demo
                view_matrix,
                proj_matrix,
                renderer=p.ER_BULLET_HARDWARE_OPENGL
            )
            
            print(f"  ✅ {name}: Captured {images[0]}x{images[1]} RGB-D image")
        except Exception as e:
            print(f"  ⚠️ Camera {name} failed: {e}")
    
    print("🎬 Running interactive simulation...")
    print("You should see:")
    print("- 3 stable wooden shelves (they won't fall!)")
    print("- Kuka robot arm")
    print("- Colorful objects on shelf surfaces")
    print("- Robot will move periodically")
    print()
    print("GUI controls:")
    print("- Drag with left mouse to rotate view")  
    print("- Drag with right mouse to pan")
    print("- Scroll to zoom")
    print("- Press 'q' to quit or close the window")
    
    # Interactive simulation loop with robot movement
    start_time = time.time()
    step_count = 0
    robot_targets = [
        (0.3, -0.3, 0.8),   # Above first shelf
        (-0.2, -0.5, 0.6),  # Above second shelf
        (0.1, -0.7, 0.4),   # Above third shelf
        (0.0, -0.8, 0.8),   # Back to home
    ]
    
    try:
        while time.time() - start_time < 45:  # Run for 45 seconds
            p.stepSimulation()
            step_count += 1
            
            # Move robot to demonstrate control
            if step_count % 300 == 0:  # Every 5 seconds
                target_index = (step_count // 300 - 1) % len(robot_targets)
                target = robot_targets[target_index]
                
                print(f"🔧 Moving robot to shelf {target_index + 1}: {target}")
                success = robot.move_to_position(target)
                
                if success:
                    current_pos = robot.get_end_effector_position()
                    print(f"  ✅ Robot moved from {current_pos} to {target}")
                else:
                    print(f"  ⚠️ Robot movement failed")
            
            # Check for GUI window closure
            keys = p.getKeyboardEvents()
            if ord('q') in keys and keys[ord('q')] & p.KEY_WAS_TRIGGERED:
                print("Quit key pressed")
                break
                
            time.sleep(1/60)  # 60Hz simulation
            
    except KeyboardInterrupt:
        print("\nDemo stopped by user")
    
    print("🧹 Cleaning up...")
    robot.cleanup()
    env.cleanup()  # Won't disconnect since it doesn't own the client
    p.disconnect()
    print("✅ Demo complete!")


def run_simple_test():
    """Minimal test to verify GUI works"""
    print("🧪 Simple GUI Test...")
    
    p.connect(p.GUI)
    p.setGravity(0, 0, -9.81)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    
    # Load and step immediately
    p.loadURDF("plane.urdf")
    p.loadURDF("cube.urdf", [0, 0, 1])
    p.stepSimulation()  # Critical for macOS!
    
    print("✅ GUI should be responsive. Running for 5 seconds...")
    
    # Simple simulation
    for _ in range(300):  # 5 seconds at 60Hz
        p.stepSimulation()
        time.sleep(1/60)
    
    p.disconnect()
    print("✅ Simple test complete!")


def main():
    """Demo selector with working GUI handling"""
    print("🎮 WORKING VISUAL DEMO")
    print("=" * 30)
    print("This demo fixes the macOS GUI freezing issue!")
    print()
    print("Choose option:")
    print("1. Full demo (shelves + robot + objects)")
    print("2. Simple test (just cube falling)")
    print("3. Skip visual demo")
    
    try:
        choice = input("\nEnter choice (1/2/3): ").strip()
        
        if choice == "1":
            run_working_demo()
        elif choice == "2":
            run_simple_test()
        else:
            print("Visual demo skipped")
            
    except KeyboardInterrupt:
        print("\nDemo cancelled")
    except Exception as e:
        print(f"❌ Demo error: {e}")
        print("\nThe issue might be:")
        print("- PyBullet not installed properly")
        print("- macOS permissions for GUI access")
        print("- Try the headless demo instead:")
        print("  python src/simple_demo.py  # Choose option 2")


if __name__ == "__main__":
    main()
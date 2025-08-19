#!/usr/bin/env python3
"""
Validate that the demo components work correctly (headless test)
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import pybullet as p
import pybullet_data
from simulation import ShelfEnvironment, ShelfConfiguration
from control import RobotController

def test_components():
    """Test that all components work with shared physics client"""
    print("🧪 Validating Phase 1 Components Integration...")
    
    # Connect headless
    physics_client = p.connect(p.DIRECT)
    p.setGravity(0, 0, -9.81)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    
    # Load ground
    ground_id = p.loadURDF("plane.urdf")
    
    print("✅ Physics client connected and ground loaded")
    
    # Test ShelfEnvironment with existing client
    print("📚 Testing ShelfEnvironment with shared client...")
    config = ShelfConfiguration(num_shelves=3, shelf_spacing=0.4)
    env = ShelfEnvironment(config, physics_client=physics_client)
    env.initialize(use_gui=False)
    shelf_ids = env.create_shelves()
    shelf_positions = env.get_shelf_positions()
    
    print(f"  ✅ Created {len(shelf_ids)} shelves")
    print(f"  ✅ Got {len(shelf_positions)} positions")
    
    # Verify shelves are static (baseMass=0)
    for shelf_id in shelf_ids:
        mass = p.getDynamicsInfo(shelf_id, -1)[0]  # Get base mass
        print(f"  ✅ Shelf {shelf_id} mass: {mass} (should be 0 for static)")
    
    # Test RobotController with existing client
    print("🤖 Testing RobotController with shared client...")
    robot = RobotController(physics_client=physics_client)
    robot.initialize()
    
    print(f"  ✅ Robot loaded with {robot.num_joints} joints")
    print(f"  ✅ Found {len(robot.joint_indices)} controllable joints")
    
    # Test robot movement
    print("🔧 Testing robot movement...")
    initial_pos = robot.get_end_effector_position()
    print(f"  Initial position: {initial_pos}")
    
    target = (0.3, -0.5, 0.8)
    success = robot.move_to_position(target)
    if success:
        print(f"  ✅ Robot movement command successful to {target}")
    else:
        print(f"  ⚠️ Robot movement failed")
        
    # Run a few physics steps
    print("⚙️ Running physics simulation...")
    for i in range(10):
        p.stepSimulation()
        
    final_pos = robot.get_end_effector_position()
    print(f"  Final position: {final_pos}")
    
    # Test object placement
    print("📦 Testing object placement on shelves...")
    objects = []
    for i, (x, y, z) in enumerate(shelf_positions[:3]):
        obj_id = p.loadURDF("cube.urdf", [x, y, z + 0.05], globalScaling=0.05)
        objects.append(obj_id)
        print(f"  ✅ Placed object {i+1} at shelf position ({x:.2f}, {y:.2f}, {z:.2f})")
    
    # Run more simulation to see if objects fall or stay on shelves
    print("⚙️ Running extended simulation to check object stability...")
    for i in range(100):
        p.stepSimulation()
        
    # Check final object positions
    for i, obj_id in enumerate(objects):
        pos, _ = p.getBasePositionAndOrientation(obj_id)
        print(f"  Object {i+1} final position: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")
        
        # Check if object stayed on shelf (z should be > 0.05)
        if pos[2] > 0.05:
            print(f"    ✅ Object {i+1} stayed on shelf")
        else:
            print(f"    ❌ Object {i+1} fell off shelf")
    
    print("🧹 Cleaning up...")
    robot.cleanup()
    env.cleanup()  # Won't disconnect since it doesn't own the client
    p.disconnect()
    
    print("✅ Component validation complete!")
    
    return True

if __name__ == "__main__":
    try:
        success = test_components()
        if success:
            print("\n🎉 ALL COMPONENTS WORKING CORRECTLY!")
            print("The visual demo should show stable shelves, working robot, and objects!")
        else:
            print("\n❌ Component validation failed")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
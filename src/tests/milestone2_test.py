#!/usr/bin/env python3
"""
Milestone 2 Checkpoint - Environment stable with 10 objects
Part of Phase 1: Foundation - Milestone 2

"""

import time
import pybullet as p
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simulation import ShelfEnvironment, ShelfConfiguration


def test_environment_stability() -> bool:
    """Test 10 objects stay stable on shelves"""
    print("🚀 Testing environment stability with 10 objects...")
    
    # Setup environment
    config = ShelfConfiguration(num_shelves=2)  # 2 shelves for more space
    env = ShelfEnvironment(config)
    env.initialize(use_gui=False)
    
    # Create shelves
    shelf_ids = env.create_shelves()
    shelf_positions = env.get_shelf_positions()
    
    # Create 10 simple test objects
    objects = []
    for i in range(10):
        # Distribute objects across shelves
        shelf_idx = i % len(shelf_positions)
        x_offset = -0.2 + (i % 5) * 0.1  # 5 objects per shelf
        
        position = [
            x_offset,
            0.0,
            shelf_positions[shelf_idx][2] + 0.02  # Slightly above shelf
        ]
        
        obj_id = p.createMultiBody(
            baseMass=0.1,
            baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.02, 0.02, 0.02]),
            baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=[0.02, 0.02, 0.02]),
            basePosition=position
        )
        objects.append(obj_id)
    
    # Run simulation for stability test
    print("🔄 Running stability simulation...")
    initial_positions = []
    for obj_id in objects:
        pos, _ = p.getBasePositionAndOrientation(obj_id)
        initial_positions.append(pos)
    
    # Simulate for 2 seconds
    for _ in range(480):  # 2 seconds at 240Hz
        p.stepSimulation()
    
    # Check if objects are still stable
    stable_count = 0
    for i, obj_id in enumerate(objects):
        pos, _ = p.getBasePositionAndOrientation(obj_id)
        initial_pos = initial_positions[i]
        
        # Check if object hasn't moved too much (especially z-axis)
        if abs(pos[2] - initial_pos[2]) < 0.1:  # Within 10cm of original height
            stable_count += 1
    
    env.cleanup()
    
    success = stable_count >= 8  # 80% of objects should remain stable
    print(f"✅ Stable objects: {stable_count}/10 ({'PASS' if success else 'FAIL'})")
    
    return success


def main():
    """KISS: Simple checkpoint test"""
    success = test_environment_stability()
    
    if success:
        print("🎉 MILESTONE 2 COMPLETE - Environment stable!")
    else:
        print("❌ Milestone 2 failed - stability issues")
        
    return success


if __name__ == "__main__":
    main()
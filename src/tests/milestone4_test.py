#!/usr/bin/env python3
"""
Milestone 4 Test - Performance-Optimized Object Library
Part of Phase 1: Foundation - Milestone 4
"""

import time
import pybullet as p
import pybullet_data
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from objects import ObjectLibrary


def test_basic_object_loading() -> bool:
    """Test basic object loading functionality"""
    print("🚀 Testing basic object loading...")
    
    # Initialize PyBullet
    physics_client = p.connect(p.DIRECT)
    p.setGravity(0, 0, -9.81)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF("plane.urdf")
    
    # Initialize library
    assets_path = "assets"
    library = ObjectLibrary(assets_path)
    library.scan_objects()
    
    total_objects = library.get_object_count()
    print(f"  Found {total_objects} objects")
    
    if total_objects == 0:
        print("  ❌ No objects found")
        p.disconnect()
        return False
        
    # Test progressive loading sets
    set_5, set_20, all_objects = library.get_progressive_sets()
    print(f"  Progressive sets: {len(set_5)} → {len(set_20)} → {len(all_objects)}")
    
    # Test loading 5 objects
    start_time = time.time()
    success = library.load_objects(set_5, physics_client)
    load_time = time.time() - start_time
    
    loaded_count = library.get_loaded_count()
    print(f"  Loaded {loaded_count}/5 objects in {load_time:.3f}s")
    
    # Basic physics test
    for _ in range(50):
        p.stepSimulation()
        
    # Check objects are stable (above ground)
    stable_count = 0
    for bullet_id in library.loaded_objects.values():
        pos, _ = p.getBasePositionAndOrientation(bullet_id)
        if pos[2] > -0.1:
            stable_count += 1
    
    print(f"  Physics stability: {stable_count}/{loaded_count} objects stable")
    
    # Cleanup
    library.unload_all()
    p.disconnect()
    
    success_criteria = loaded_count >= 3 and stable_count >= loaded_count // 2
    print(f"  ✅ Object library working" if success_criteria else "  ❌ Issues detected")
    return success_criteria


def main():
    """Run Milestone 4 test"""
    print("🚀 MILESTONE 4: PERFORMANCE-OPTIMIZED OBJECT LIBRARY")
    print("=" * 50)
    
    success = test_basic_object_loading()
    
    if success:
        print("\n🎉 MILESTONE 4 COMPLETE!")
        print("✅ Object library working")
    else:
        print("\n❌ Milestone 4 failed")
        
    return success


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
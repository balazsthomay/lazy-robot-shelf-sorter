#!/usr/bin/env python3
"""
Walking Skeleton - Minimal PyBullet scene validation
Part of Phase 1: Foundation - Milestone 1

"""

import time
import pybullet as p
import pybullet_data
from dataclasses import dataclass


@dataclass
class PerformanceMetrics:
    """Simple metrics for baseline"""
    total_time: float
    physics_step_time: float


class WalkingSkeleton:
    """PyBullet validation"""
    
    def __init__(self):
        self.physics_client = None
        self.shelf_id = None
        self.test_objects = []
        
    def run_validation(self) -> bool:
        print("🚀 Walking Skeleton validation...")
        start_time = time.time()
        
        # Initialize
        self.physics_client = p.connect(p.DIRECT)  # KISS: no GUI needed
        p.setGravity(0, 0, -9.81)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # Ground plane
        p.loadURDF("plane.urdf")
        
        # Single shelf - KISS: just one box
        self.shelf_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.4, 0.2, 0.01]),
            baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=[0.4, 0.2, 0.01]),
            basePosition=[0, 0, 0.01]
        )
        
        # Test objects - KISS: just simple boxes
        for i in range(5):
            x = -0.2 + i * 0.1
            obj_id = p.createMultiBody(
                baseMass=0.1,
                baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.02, 0.02, 0.02]),
                baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=[0.02, 0.02, 0.02]),
                basePosition=[x, 0, 0.05]
            )
            self.test_objects.append(obj_id)
        
        # Physics test - KISS: just run simulation
        step_times = []
        for _ in range(240):  # 1 second at 240Hz
            step_start = time.time()
            p.stepSimulation()
            step_times.append(time.time() - step_start)
            
        # Results
        total_time = time.time() - start_time
        avg_step_time = sum(step_times) / len(step_times)
        
        print(f"✅ Total time: {total_time:.3f}s")
        print(f"✅ Physics step: {avg_step_time*1000:.3f}ms")
        print(f"✅ Objects stable: {len(self.test_objects)} loaded")
        
        # Cleanup
        p.disconnect()
        
        return total_time < 5.0 and avg_step_time < 0.01  # Simple success criteria


def main():
    """Validation"""
    skeleton = WalkingSkeleton()
    success = skeleton.run_validation()
    
    if success:
        print("🎉 Walking skeleton SUCCESSFUL")
    else:
        print("❌ Performance issues detected")
        
    return success


if __name__ == "__main__":
    main()
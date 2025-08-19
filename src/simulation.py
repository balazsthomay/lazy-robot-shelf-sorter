#!/usr/bin/env python3
"""
Simulation Environment - Core PyBullet environment management
Part of Phase 1: Foundation - Milestone 2

"""

import pybullet as p
import pybullet_data
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class ShelfConfiguration:
    """Shelf configuration"""
    num_shelves: int = 1
    shelf_width: float = 0.8  # 80cm
    shelf_depth: float = 0.4  # 40cm
    shelf_height: float = 0.3  # 30cm
    shelf_spacing: float = 0.4  # Vertical spacing


class ShelfEnvironment:
    """Manage shelf environment in PyBullet"""
    
    def __init__(self, config: ShelfConfiguration, physics_client: int = None):
        self.config = config
        self.physics_client = physics_client
        self.shelf_ids: List[int] = []
        self.ground_id = None
        self.owns_physics_client = physics_client is None
        
    def initialize(self, use_gui: bool = False) -> None:
        """Initialization"""
        if self.physics_client is None:
            if use_gui:
                self.physics_client = p.connect(p.GUI)
            else:
                self.physics_client = p.connect(p.DIRECT)
                
            p.setGravity(0, 0, -9.81)
            p.setTimeStep(1./240.)
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            
            # Ground plane - only create if we own the connection
            self.ground_id = p.loadURDF("plane.urdf")
        # If physics_client was provided, assume environment is already set up
        
    def create_shelves(self) -> List[int]:
        """Create box shelves"""
        self.shelf_ids = []
        
        for i in range(self.config.num_shelves):
            # Calculate shelf position
            shelf_z = i * self.config.shelf_spacing + 0.01
            
            # Create shelf as simple box
            shelf_id = p.createMultiBody(
                baseMass=0,  # Static
                baseCollisionShapeIndex=p.createCollisionShape(
                    p.GEOM_BOX, 
                    halfExtents=[
                        self.config.shelf_width / 2,
                        self.config.shelf_depth / 2,
                        0.01  # Thin shelf
                    ]
                ),
                baseVisualShapeIndex=p.createVisualShape(
                    p.GEOM_BOX,
                    halfExtents=[
                        self.config.shelf_width / 2,
                        self.config.shelf_depth / 2,
                        0.01
                    ],
                    rgbaColor=[0.8, 0.6, 0.4, 1.0]  # Wood color
                ),
                basePosition=[0, 0, shelf_z]
            )
            
            self.shelf_ids.append(shelf_id)
            
        return self.shelf_ids
        
    def get_shelf_positions(self) -> List[Tuple[float, float, float]]:
        """Position list for now"""
        positions = []
        for i in range(self.config.num_shelves):
            shelf_z = i * self.config.shelf_spacing + 0.02  # Slightly above shelf
            positions.append((0.0, 0.0, shelf_z))
        return positions
        
    def cleanup(self) -> None:
        """Cleanup"""
        if self.physics_client is not None and self.owns_physics_client:
            p.disconnect(self.physics_client)
            self.physics_client = None


def main():
    """Test function"""
    print("🚀 Testing ShelfEnvironment...")
    
    # Test with different configurations
    configs = [
        ShelfConfiguration(num_shelves=1),
        ShelfConfiguration(num_shelves=2), 
        ShelfConfiguration(num_shelves=3)
    ]
    
    for i, config in enumerate(configs):
        print(f"Testing {config.num_shelves} shelf configuration...")
        
        env = ShelfEnvironment(config)
        env.initialize(use_gui=False)
        shelf_ids = env.create_shelves()
        positions = env.get_shelf_positions()
        
        print(f"  ✅ Created {len(shelf_ids)} shelves")
        print(f"  ✅ Got {len(positions)} positions")
        
        env.cleanup()
        
    print("🎉 ShelfEnvironment tests passed!")


if __name__ == "__main__":
    main()
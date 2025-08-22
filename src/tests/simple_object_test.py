#!/usr/bin/env python3
"""
Simple Object Loading Test - Phase 2 Validation
Loads 6 objects (3 YCB + 3 GSO) in PyBullet GUI without shelves
Focus: Minimal, working object loading validation
"""

import os
import sys
import time
import pybullet as p
import pybullet_data
from pathlib import Path

# Add src directory to path for imports
src_path = Path(__file__).parent.parent
sys.path.append(str(src_path))

from objects import ObjectLibrary


class SimpleObjectValidator:
    """Minimal object loading validator for Phase 2"""
    
    def __init__(self):
        self.physics_client = None
        self.object_library = None
        self.loaded_objects = []
        
    def setup_physics(self) -> bool:
        """Setup single PyBullet physics client with GUI"""
        print("🔧 Setting up PyBullet physics...")
        
        # Single physics client connection
        self.physics_client = p.connect(p.GUI)
        if self.physics_client < 0:
            print("❌ Failed to connect to PyBullet GUI")
            return False
            
        # Basic physics setup with consistent client ID
        p.setGravity(0, 0, -9.81, physicsClientId=self.physics_client)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # Load ground plane
        p.loadURDF("plane.urdf", physicsClientId=self.physics_client)
        
        # Set camera for optimal object viewing
        p.resetDebugVisualizerCamera(
            cameraDistance=2.0,
            cameraYaw=30,
            cameraPitch=-20,
            cameraTargetPosition=[1.5, 0, 0.2],
            physicsClientId=self.physics_client
        )
        
        print("✅ Physics setup complete")
        print("   Navigation: CTRL + DRAG to move camera (only method that works)")
        return True
        
    def select_objects(self) -> tuple:
        """Select 3 YCB + 3 GSO objects that are likely to work"""
        print("\n📦 Scanning object library...")
        
        # Initialize and scan object library
        assets_path = os.path.join(os.path.dirname(__file__), '..', '..', 'assets')
        self.object_library = ObjectLibrary(assets_path)
        self.object_library.physics_client = self.physics_client
        self.object_library.scan_objects()
        
        print(f"   Found {self.object_library.get_object_count()} total objects")
        
        # Get all available objects
        all_metadata = self.object_library.metadata
        ycb_objects = [obj_id for obj_id in all_metadata.keys() if obj_id.startswith('ycb_')]
        gso_objects = [obj_id for obj_id in all_metadata.keys() if obj_id.startswith('gso_')]
        
        print(f"   YCB objects: {len(ycb_objects)}")
        print(f"   GSO objects: {len(gso_objects)}")
        
        
        # Select 3 YCB objects (prefer simple shapes)
        ycb_priority = ['ycb_025_mug', 'ycb_013_apple', 'ycb_011_banana']
        selected_ycb = []
        for obj in ycb_priority:
            if obj in ycb_objects:
                selected_ycb.append(obj)
            if len(selected_ycb) >= 3:
                break
                
        # Fill remaining YCB slots if needed
        while len(selected_ycb) < 3 and len(ycb_objects) > len(selected_ycb):
            for obj in ycb_objects:
                if obj not in selected_ycb:
                    selected_ycb.append(obj)
                    break
                    
        # Select 3 GSO objects (prefer simple containers)
        gso_priority = ['gso_Threshold_Porcelain_Coffee_Mug_All_Over_Bead_White',
                       'gso_ACE_Coffee_Mug_Kristen_16_oz_cup',
                       'gso_Threshold_Porcelain_Teapot_White']
        selected_gso = []
        for obj in gso_priority:
            if obj in gso_objects:
                selected_gso.append(obj)
            if len(selected_gso) >= 3:
                break
                
        # Fill remaining GSO slots if needed
        while len(selected_gso) < 3 and len(gso_objects) > len(selected_gso):
            for obj in gso_objects:
                if obj not in selected_gso:
                    selected_gso.append(obj)
                    break
                    
        print(f"\n   Selected YCB objects: {selected_ycb}")
        print(f"   Selected GSO objects: {selected_gso}")
        
        return selected_ycb, selected_gso
        
    def load_and_place_objects(self, ycb_objects: list, gso_objects: list) -> bool:
        """Load objects using ObjectLibrary and position them in a line"""
        print("\n🎯 Loading and placing objects...")
        
        all_objects = ycb_objects + gso_objects
        print(f"   Loading {len(all_objects)} objects total")
        
        # Load objects through ObjectLibrary
        success = self.object_library.load_objects(all_objects, self.physics_client)
        if not success:
            print("❌ Failed to load objects through ObjectLibrary")
            return False
            
        loaded_count = len(self.object_library.loaded_objects)
        print(f"   ObjectLibrary loaded {loaded_count} objects")
        
        # Debug: Show which objects actually loaded
        loaded_ids = list(self.object_library.loaded_objects.keys())
        failed_ids = [obj for obj in all_objects if obj not in loaded_ids]
        print(f"   Successfully loaded: {loaded_ids}")
        if failed_ids:
            print(f"   Failed to load: {failed_ids}")
        
        # Position objects in a line
        x_position = 0.0
        spacing = 0.5
        
        for i, obj_id in enumerate(all_objects):
            if obj_id in self.object_library.loaded_objects:
                bullet_id = self.object_library.loaded_objects[obj_id]
                position = (x_position, 0.0, 0.1)
                
                # Position object with consistent physics client
                p.resetBasePositionAndOrientation(
                    bullet_id, 
                    position, 
                    (0, 0, 0, 1), 
                    physicsClientId=self.physics_client
                )
                
                self.loaded_objects.append((obj_id, bullet_id, position))
                print(f"      Placed {obj_id} at x={x_position:.1f}")
                x_position += spacing
                
        print(f"✅ Successfully placed {len(self.loaded_objects)} objects")
        
        # Let physics settle
        for _ in range(60):
            p.stepSimulation(physicsClientId=self.physics_client)
            time.sleep(1./60.)
            
        return len(self.loaded_objects) > 0
        
    def run_validation(self) -> None:
        """Run complete validation with user interaction"""
        print("=" * 50)
        print("SIMPLE OBJECT LOADING TEST - PHASE 2")
        print("=" * 50)
        
        # Setup physics
        if not self.setup_physics():
            print("❌ Physics setup failed")
            return
            
        # Select objects
        try:
            ycb_objects, gso_objects = self.select_objects()
        except Exception as e:
            print(f"❌ Object selection failed: {e}")
            return
            
        # Load and place objects
        if not self.load_and_place_objects(ycb_objects, gso_objects):
            print("❌ Object loading failed")
            return
            
        # Validation complete
        print("\n🎉 VALIDATION COMPLETE!")
        print(f"   Objects loaded: {len(self.loaded_objects)}")
        print(f"   YCB objects: {len([obj for obj in ycb_objects if any(obj == o[0] for o in self.loaded_objects)])}")
        print(f"   GSO objects: {len([obj for obj in gso_objects if any(obj == o[0] for o in self.loaded_objects)])}")
        
        print("\n👀 WHAT YOU SHOULD SEE:")
        print("   - Ground plane with objects in a line")
        print("   - Objects in their natural colors and textures")
        print("   - 6 objects total (3 YCB + 3 GSO)")
        print("   - Objects spaced 0.5m apart along x-axis")
        
        print("\n🎮 CONTROLS:")
        print("   - CTRL + DRAG: Move camera (only working method)")
        print("   - Close window or press 'q' to exit")
        
        # Keep GUI open for inspection
        print("\n⏳ GUI running... Close window or press 'q' to exit")
        
        try:
            while True:
                p.stepSimulation(physicsClientId=self.physics_client)
                
                # Check for exit conditions
                keys = p.getKeyboardEvents(physicsClientId=self.physics_client)
                if ord('q') in keys and keys[ord('q')] & p.KEY_WAS_TRIGGERED:
                    print("Exit key pressed")
                    break
                    
                time.sleep(1./60.)
                
        except KeyboardInterrupt:
            print("\nValidation stopped by user")
        except Exception as e:
            print(f"\nValidation ended: {e}")
        finally:
            p.disconnect(self.physics_client)
            print("✅ Validation complete")


def main():
    """Main entry point"""
    validator = SimpleObjectValidator()
    validator.run_validation()


if __name__ == "__main__":
    main()
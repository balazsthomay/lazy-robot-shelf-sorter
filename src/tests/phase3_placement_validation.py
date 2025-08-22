#!/usr/bin/env python3
"""
Phase 3 Placement Logic Validation
Tests placement commands, spatial constraints, and strategy implementation
"""

import os
import sys
import time
import numpy as np
import pybullet as p
import pybullet_data
from pathlib import Path

# Add src directory to path for imports
src_path = Path(__file__).parent.parent
sys.path.append(str(src_path))

from objects import ObjectLibrary
from simulation import ShelfEnvironment, ShelfConfiguration
from vision import DinoModel, CameraSystem, CameraConfiguration, ShelfZoneManager, Camera
from placement import PlacementEngine, SimilarityEngine, PlacementStrategy, PlacementCommand, PlacementResult
from spatial import ShelfSpaceManager, Rectangle
# Note: Integration layer validation will be in separate script


class Phase3PlacementValidator:
    """Validates Phase 3 placement logic with visual GUI demonstration"""
    
    def __init__(self):
        self.physics_client = None
        self.object_library = None
        self.shelf_env = None
        self.dino_model = None
        self.zone_manager = None
        self.similarity_engine = None
        self.placement_engine = None
        self.space_manager = None
        self.loaded_objects = []
        self.placement_commands = []
        
    def setup_environment(self) -> bool:
        """Setup PyBullet environment with shelves"""
        print("🔧 Setting up environment...")
        
        # Physics setup
        self.physics_client = p.connect(p.GUI)
        if self.physics_client < 0:
            print("❌ Failed to connect to PyBullet GUI")
            return False
            
        p.setGravity(0, 0, -9.81, physicsClientId=self.physics_client)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF("plane.urdf", physicsClientId=self.physics_client)
        p.stepSimulation(physicsClientId=self.physics_client)
        
        # Camera setup for good viewing of placement decisions
        p.resetDebugVisualizerCamera(
            cameraDistance=2.2,
            cameraYaw=45,
            cameraPitch=-25,
            cameraTargetPosition=[0, 0, 0.5],
            physicsClientId=self.physics_client
        )
        
        # Create shelf environment
        config = ShelfConfiguration(num_shelves=3, shelf_spacing=0.4)
        self.shelf_env = ShelfEnvironment(config, physics_client=self.physics_client)
        self.shelf_env.initialize(use_gui=False)
        shelf_ids = self.shelf_env.create_shelves()
        
        print(f"  ✅ Created {len(shelf_ids)} shelves for placement testing")
        
        # Step simulation to update display
        for _ in range(10):
            p.stepSimulation(physicsClientId=self.physics_client)
            time.sleep(0.01)
            
        return True
        
    def setup_placement_system(self) -> bool:
        """Initialize complete Phase 2+3 pipeline"""
        print("🎯 Setting up placement system...")
        
        try:
            # Initialize DINO model
            self.dino_model = DinoModel()
            self.dino_model.initialize()
            
            # Setup zone manager for 18 zones (3 shelves × 6 zones each)
            self.zone_manager = ShelfZoneManager()
            self.zone_manager.initialize()
            
            # Initialize similarity engine
            self.similarity_engine = SimilarityEngine()
            self.similarity_engine.initialize()
            
            # Setup spatial manager
            self.space_manager = ShelfSpaceManager()
            self.space_manager.initialize()
            
            # Define zone boundaries for spatial management
            zones = self.zone_manager.get_all_zones()
            for zone_id, (x, y, z) in zones.items():
                # Each zone is a 60cm x 60cm rectangle (large enough for 40cm objects)
                bounds = Rectangle(x - 0.3, y - 0.3, 0.6, 0.6)
                self.space_manager.set_zone_bounds(zone_id, bounds)
            
            # Initialize placement engine
            self.placement_engine = PlacementEngine(
                similarity_engine=self.similarity_engine
            )
            self.placement_engine.initialize()
            
            # Setup placement engine's space manager with our zone bounds
            for zone_id, bounds in [(zone_id, Rectangle(x - 0.3, y - 0.3, 0.6, 0.6)) 
                                  for zone_id, (x, y, z) in zones.items()]:
                self.placement_engine.space_manager.set_zone_bounds(zone_id, bounds)
            
            # Phase 3 placement system ready
            
            print(f"  ✅ Placement system ready with {len(zones)} zones")
            return True
            
        except Exception as e:
            print(f"❌ Placement system setup failed: {e}")
            return False
        
    def load_test_objects(self) -> bool:
        """Load objects for placement testing"""
        print("📦 Loading objects for placement testing...")
        
        # Initialize object library
        assets_path = os.path.join(os.path.dirname(__file__), '..', '..', 'assets')
        self.object_library = ObjectLibrary(assets_path)
        self.object_library.physics_client = self.physics_client
        self.object_library.scan_objects()
        
        # Select diverse test objects for different strategies
        test_objects = [
            'ycb_015_peach',     # For PLACE strategy
            'ycb_025_mug',       # For STACK strategy  
            'ycb_033_spatula',   # For LEAN strategy
        ]
        
        # Add GSO objects if available
        all_metadata = self.object_library.metadata
        gso_objects = [obj_id for obj_id in all_metadata.keys() if obj_id.startswith('gso_')]
        test_objects.extend(gso_objects[:3])  # GROUP strategy
        
        # Load objects (initially positioned off-screen)
        success = self.object_library.load_objects(test_objects, self.physics_client)
        if not success:
            print("❌ Failed to load test objects")
            return False
        
        # Position objects in "incoming area" (before placement)
        loaded_objects = list(self.object_library.loaded_objects.keys())
        incoming_positions = [
            (-1.0, -0.5, 0.1),   # Incoming object area
            (-1.0, -0.3, 0.1),
            (-1.0, -0.1, 0.1),
            (-1.0, 0.1, 0.1),
            (-1.0, 0.3, 0.1),
            (-1.0, 0.5, 0.1),
        ]
        
        for i, obj_id in enumerate(loaded_objects):
            if i < len(incoming_positions):
                position = incoming_positions[i]
                bullet_id = self.object_library.loaded_objects[obj_id]
                
                p.resetBasePositionAndOrientation(
                    bullet_id, 
                    position, 
                    (0, 0, 0, 1), 
                    physicsClientId=self.physics_client
                )
                
                self.loaded_objects.append((obj_id, bullet_id, position))
                print(f"  ✅ Loaded {obj_id} in incoming area")
        
        # Let physics settle
        for _ in range(30):
            p.stepSimulation(physicsClientId=self.physics_client)
            time.sleep(1./60.)
            
        return len(self.loaded_objects) > 0
        
    def extract_features_for_placement(self) -> bool:
        """Extract features needed for placement decisions"""
        print("🔍 Extracting features for placement...")
        
        try:
            # Extract features for zones (simplified)
            zones = self.zone_manager.get_all_zones()
            for zone_id, (x, y, z) in zones.items():
                # Generate zone features (simplified for demo)
                zone_features = np.random.rand(768)
                zone_features = zone_features / np.linalg.norm(zone_features)
                self.similarity_engine.store_zone_embedding(zone_id, zone_features)
                
            # Extract features for objects
            for obj_id, bullet_id, position in self.loaded_objects:
                # Generate object features (simplified for demo)
                object_features = np.random.rand(768)
                object_features = object_features / np.linalg.norm(object_features)
                
                # Ensure 1D embedding
                object_features = object_features.flatten()
                
                self.similarity_engine.store_object_embedding(obj_id, object_features)
                
            print(f"  ✅ Object embeddings stored: {len(self.similarity_engine.object_embeddings)}")
            print(f"  ✅ Zone embeddings stored: {len(self.similarity_engine.zone_embeddings)}")
            
            # Test similarity computation
            test_obj_id = list(self.similarity_engine.object_embeddings.keys())[0]
            test_embedding = self.similarity_engine.object_embeddings[test_obj_id]
            heatmap = self.similarity_engine.get_similarity_heatmap(test_embedding)
            print(f"  ✅ Test heatmap computed: {len(heatmap)} scores")
                
            print(f"  ✅ Features extracted for {len(zones)} zones and {len(self.loaded_objects)} objects")
            return True
            
        except Exception as e:
            print(f"❌ Feature extraction failed: {e}")
            return False
    
    def demonstrate_placement_strategies(self) -> bool:
        """Demonstrate all 4 placement strategies with GUI visualization"""
        print("🎲 Demonstrating placement strategies...")
        
        strategies = [
            (PlacementStrategy.PLACE, "Standard shelf placement"),
            (PlacementStrategy.STACK, "Stack on existing objects"),
            (PlacementStrategy.LEAN, "Lean against back wall"),
            (PlacementStrategy.GROUP, "Group with similar items")
        ]
        
        for i, (obj_id, bullet_id, current_pos) in enumerate(self.loaded_objects[:4]):
            if i >= len(strategies):
                break
                
            strategy, description = strategies[i]
            
            print(f"\n  📍 Object {i+1}: {obj_id}")
            print(f"     Strategy: {strategy.value} - {description}")
            
            # Get object metadata for size
            metadata = self.object_library.get_metadata(obj_id)
            object_size = (0.1, 0.1) if not metadata else (metadata.estimated_size, metadata.estimated_size)
            
            # Get object embedding
            object_embedding = self.similarity_engine.object_embeddings.get(obj_id)
            
            if object_embedding is None:
                print(f"     ❌ No embedding found for {obj_id}")
                continue
                
            
            # Use the FULL placement engine as intended for Phase 3 validation
            try:
                print(f"     🔍 Using placement engine to find placement...")
                print(f"     🔍 Object embedding shape: {object_embedding.shape}")
                print(f"     🔍 Object size: {object_size}")
                print(f"     🔍 Strategy: {strategy}")
                
                # This tests the complete Phase 3 pipeline:
                # 1. SimilarityEngine.get_similarity_heatmap()
                # 2. CandidateGenerator.generate() 
                # 3. Strategy application
                # 4. BasicConstraintChecker.is_valid_position()
                # 5. PlacementScorer.score_placement()
                print(f"     🔍 About to call placement engine...")
                result = self.placement_engine.find_placement(
                    obj_id, object_embedding, object_size, strategy
                )
                print(f"     🔍 Placement engine returned: {result.success}")
                
                if result.success and result.placement_command:
                    cmd = result.placement_command
                    
                    print(f"     ✅ Placement found!")
                    print(f"        Zone: {cmd.zone_id}")
                    print(f"        Position: ({cmd.position[0]:.2f}, {cmd.position[1]:.2f}, {cmd.position[2]:.2f})")
                    print(f"        Confidence: {cmd.confidence_score:.3f}")
                    
                    # Move object to placement position
                    p.resetBasePositionAndOrientation(
                        bullet_id,
                        cmd.position,
                        cmd.orientation,
                        physicsClientId=self.physics_client
                    )
                    
                    # Update space manager with occupied space
                    occupied_rect = Rectangle(
                        cmd.position[0] - object_size[0]/2,
                        cmd.position[1] - object_size[1]/2,
                        object_size[0],
                        object_size[1]
                    )
                    self.space_manager.add_occupied_space(cmd.zone_id, occupied_rect)
                    
                    self.placement_commands.append(cmd)
                    
                    # Let physics settle and show result
                    for _ in range(60):
                        p.stepSimulation(physicsClientId=self.physics_client)
                        time.sleep(1./30.)  # Slower for visual effect
                        
                else:
                    print(f"     ❌ Placement failed: {result.failure_reason}")
                    
            except Exception as e:
                print(f"     ❌ Placement error: {e}")
                
        return len(self.placement_commands) > 0
    
    def visualize_placement_analysis(self) -> None:
        """Show placement analysis and metrics"""
        print("\n📊 PLACEMENT ANALYSIS:")
        
        # Show zone efficiency
        zones = self.zone_manager.get_all_zones()
        occupied_zones = 0
        total_efficiency = 0.0
        
        for zone_id in zones.keys():
            efficiency = self.space_manager.get_zone_efficiency(zone_id)
            if efficiency > 0:
                occupied_zones += 1
                total_efficiency += efficiency
                print(f"     Zone {zone_id}: {efficiency:.1%} occupied")
        
        avg_efficiency = total_efficiency / max(occupied_zones, 1)
        print(f"     Average zone efficiency: {avg_efficiency:.1%}")
        print(f"     Zones used: {occupied_zones}/{len(zones)}")
        
        # Show placement command summary
        print(f"\n📋 PLACEMENT COMMANDS GENERATED:")
        for i, cmd in enumerate(self.placement_commands):
            strategy_used = "Unknown"
            if i < len(self.loaded_objects):
                obj_id = self.loaded_objects[i][0]
                print(f"     {i+1}. {obj_id}")
                print(f"        → Zone: {cmd.zone_id}")
                print(f"        → Position: ({cmd.position[0]:.2f}, {cmd.position[1]:.2f}, {cmd.position[2]:.2f})")
                print(f"        → Confidence: {cmd.confidence_score:.3f}")
    
    def run_validation(self) -> None:
        """Run complete Phase 3 validation"""
        print("=" * 60)
        print("PHASE 3 PLACEMENT LOGIC VALIDATION")
        print("=" * 60)
        
        try:
            # Setup environment
            if not self.setup_environment():
                print("❌ Environment setup failed")
                return
                
            # Setup placement system
            if not self.setup_placement_system():
                print("❌ Placement system setup failed")
                return
                
            # Load test objects
            if not self.load_test_objects():
                print("❌ Object loading failed")
                return
                
            # Extract features
            if not self.extract_features_for_placement():
                print("❌ Feature extraction failed")
                return
                
            print("\n⏳ Starting placement demonstration in 3 seconds...")
            for i in range(3, 0, -1):
                print(f"   {i}...")
                time.sleep(1)
            print("   Go!")
            
            # Demonstrate placement strategies
            if not self.demonstrate_placement_strategies():
                print("❌ Placement demonstration failed")
                return
                
            # Show analysis
            self.visualize_placement_analysis()
            
            # Success!
            print("\n🎉 PHASE 3 VALIDATION COMPLETE!")
            print("\n👀 WHAT YOU SHOULD SEE:")
            print("   - Objects moved from incoming area to optimal shelf positions")
            print("   - Different placement strategies demonstrated (PLACE, STACK, LEAN, GROUP)")
            print("   - Spatial constraints respected (no overlapping objects)")
            print("   - Zone efficiency optimization in action")
            
            print("\n📊 PLACEMENT SYSTEM STATUS:")
            space_state = self.space_manager.get_state()
            print(f"   - Zones defined: {space_state['num_zones']}")
            print(f"   - Objects placed: {len(self.placement_commands)}")
            print(f"   - Occupied spaces: {space_state['total_occupied_spaces']}")
            print(f"   - Average efficiency: {space_state['average_efficiency']:.1%}")
            
            print("\n🎮 CONTROLS:")
            print("   - CTRL + DRAG: Move camera to inspect placements")
            print("   - Close window or press 'q' to exit")
            
            # Keep GUI open for inspection
            print("\n⏳ GUI running... Close window or press 'q' to exit")
            
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
            print(f"\nValidation error: {e}")
        finally:
            self.cleanup()
            
    def cleanup(self) -> None:
        """Clean up all resources"""
        print("🧹 Cleaning up...")
        
        if self.object_library:
            self.object_library.unload_all()
            
        if self.shelf_env:
            self.shelf_env.cleanup()
            
        if self.dino_model:
            self.dino_model.cleanup()
            
        if self.zone_manager:
            self.zone_manager.cleanup()
            
        if self.similarity_engine:
            self.similarity_engine.cleanup()
            
        if self.space_manager:
            self.space_manager.cleanup()
            
        if self.placement_engine:
            self.placement_engine.cleanup()
            
        if self.physics_client is not None:
            p.disconnect(self.physics_client)
            
        print("✅ Cleanup complete")


def main():
    """Main entry point"""
    validator = Phase3PlacementValidator()
    validator.run_validation()


if __name__ == "__main__":
    main()
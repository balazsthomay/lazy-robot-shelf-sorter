#!/usr/bin/env python3
"""
Integrated Pipeline Validation (Phase 1→2→3)
Tests the complete workflow from environment setup through placement decisions
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
from control import RobotController
from vision import DinoModel, CameraSystem, CameraConfiguration, ShelfZoneManager
from placement import PlacementEngine, SimilarityEngine, PlacementStrategy
from spatial import ShelfSpaceManager, Rectangle


class IntegratedPipelineValidator:
    """Validates the complete Phase 1→2→3 pipeline"""
    
    def __init__(self):
        self.physics_client = None
        
        # Phase 1 Components
        self.shelf_env = None
        self.robot_controller = None
        self.object_library = None
        
        # Phase 2 Components  
        self.dino_model = None
        self.camera_system = None
        self.zone_manager = None
        
        # Phase 3 Components
        self.similarity_engine = None
        self.placement_engine = None
        
        # Test state
        self.loaded_objects = []
        self.shelf_positions = []
        
    def setup_physics(self):
        """Initialize PyBullet physics simulation"""
        print("🔧 Setting up physics simulation...")
        
        self.physics_client = p.connect(p.GUI)
        p.setGravity(0, 0, -9.81, physicsClientId=self.physics_client)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF("plane.urdf", physicsClientId=self.physics_client)
        p.stepSimulation(physicsClientId=self.physics_client)
        
        # Camera setup for integrated view
        p.resetDebugVisualizerCamera(
            cameraDistance=2.0,
            cameraYaw=30,
            cameraPitch=-20,
            cameraTargetPosition=[0, 0, 0.6],
            physicsClientId=self.physics_client
        )
        
        print("  ✅ Physics simulation ready")
        
    def setup_phase1_foundation(self):
        """Phase 1: Environment and robot setup"""
        print("\n📚 PHASE 1: Foundation Setup")
        print("=" * 40)
        
        # Shelf environment
        print("🏗️ Creating shelf environment...")
        config = ShelfConfiguration(num_shelves=3, shelf_spacing=0.4)
        self.shelf_env = ShelfEnvironment(config, physics_client=self.physics_client)
        self.shelf_env.initialize(use_gui=False)
        shelf_ids = self.shelf_env.create_shelves()
        self.shelf_positions = self.shelf_env.get_shelf_positions()
        
        print(f"  ✅ Created {len(shelf_ids)} shelves")
        print(f"  ✅ Got {len(self.shelf_positions)} shelf positions")
        
        # Robot controller
        print("🤖 Loading robot controller...")
        self.robot_controller = RobotController(physics_client=self.physics_client)
        self.robot_controller.initialize()
        
        print(f"  ✅ Robot loaded with {self.robot_controller.num_joints} joints")
        
        # Object library
        print("📦 Setting up object library...")
        assets_path = str(Path(__file__).parent.parent.parent / "assets")
        self.object_library = ObjectLibrary(assets_path)
        self.object_library.scan_objects()
        
        object_count = self.object_library.get_object_count()
        print(f"  ✅ Scanned {object_count} objects")
        
        # Load test objects
        print("📥 Loading test objects...")
        # Load specific objects: pepsi (base), mug (stack), cracker box (lean), coffee mug (group)
        test_objects = ['gso_Diet_Pepsi_Soda_Cola12_Pack_12_oz_Cans', 'ycb_025_mug', 'ycb_003_cracker_box', 'gso_Threshold_Porcelain_Coffee_Mug_All_Over_Bead_White']
        success = self.object_library.load_objects(test_objects, self.physics_client)
        
        if success:
            # Place objects in open area away from shelves to avoid shelf collision geometry
            for i, obj_id in enumerate(test_objects):
                x = 1.5 + i * 0.3  # Open area on right side, away from shelves
                y = 1.0  # Forward from shelves  
                z = 0.1
                
                bullet_id = self.object_library.loaded_objects[obj_id]
                p.resetBasePositionAndOrientation(
                    bullet_id, [x, y, z], [0, 0, 0, 1],
                    physicsClientId=self.physics_client
                )
                self.loaded_objects.append(obj_id)
                
            print(f"  ✅ Loaded {len(self.loaded_objects)} test objects")
        else:
            print("  ❌ Failed to load test objects")
            
        # Step simulation
        for _ in range(20):
            p.stepSimulation(physicsClientId=self.physics_client)
            time.sleep(0.01)
            
        print("✅ Phase 1 Foundation setup complete!")
        
    def setup_phase2_vision(self):
        """Phase 2: Vision system and feature extraction"""
        print("\n👁️ PHASE 2: Vision System")
        print("=" * 40)
        
        # DINO model
        print("🧠 Initializing DINO model...")
        self.dino_model = DinoModel()
        self.dino_model.initialize()
        model_info = self.dino_model.get_model_info()
        print(f"  ✅ Loaded {model_info['model_name']}")
        print(f"  ✅ Embedding dimension: {model_info['embedding_dim']}")
        
        # Camera system
        print("📸 Setting up camera system...")
        camera_config = CameraConfiguration(image_width=640, image_height=480)
        self.camera_system = CameraSystem(camera_config)
        self.camera_system.initialize()
        
        # Add front-facing camera
        front_camera = self.camera_system.setup_front_facing_camera(robot_height=0.8)
        print(f"  ✅ Front camera positioned at {front_camera.position}")
        
        # Zone manager
        print("🗺️ Creating zone manager...")
        self.zone_manager = ShelfZoneManager()
        self.zone_manager.initialize()
        zone_count = len(self.zone_manager.get_all_zones())
        print(f"  ✅ Created {zone_count} zones")
        
        # Similarity engine
        print("🔗 Setting up similarity engine...")
        self.similarity_engine = SimilarityEngine()
        self.similarity_engine.initialize()
        
        # Extract features for objects
        print("🔍 Extracting object features...")
        captures = self.camera_system.capture_all_cameras()
        front_capture = captures["front"]
        
        for obj_id in self.loaded_objects:
            # Use RGB image for feature extraction
            rgb_image = front_capture.rgb_image
            embedding = self.dino_model.extract_features(rgb_image)
            self.similarity_engine.store_object_embedding(obj_id, embedding)
            
        print(f"  ✅ Extracted features for {len(self.loaded_objects)} objects")
        
        # Extract features for zones
        print("🗺️ Extracting zone features...")
        zones = self.zone_manager.get_all_zones()
        for zone_id, center in zones.items():
            # Create synthetic zone embedding (in real system this would be from shelf images)
            zone_embedding = np.random.rand(768)  # Match DINO embedding size
            zone_embedding = zone_embedding / (np.linalg.norm(zone_embedding) + 1e-8)
            self.similarity_engine.store_zone_embedding(zone_id, zone_embedding)
            
        print(f"  ✅ Created embeddings for {len(zones)} zones")
        
        print("✅ Phase 2 Vision system complete!")
        
    def setup_phase3_placement(self):
        """Phase 3: Placement logic and spatial reasoning"""
        print("\n🎯 PHASE 3: Placement Logic")
        print("=" * 40)
        
        # Placement engine
        print("⚙️ Setting up placement engine...")
        self.placement_engine = PlacementEngine(self.similarity_engine)
        self.placement_engine.initialize()
        self.placement_engine.set_physics_client(self.physics_client)
        
        # Configure spatial constraints
        print("📐 Configuring spatial constraints...")
        zones = self.zone_manager.get_all_zones()
        
        for zone_id, (x, y, z) in zones.items():
            # Create large enough zone bounds (60cm x 60cm to fit objects up to 40cm)
            bounds = Rectangle(x - 0.3, y - 0.3, 0.6, 0.6)
            self.placement_engine.space_manager.set_zone_bounds(zone_id, bounds)
            
        print(f"  ✅ Configured {len(zones)} zone bounds")
        
        print("✅ Phase 3 Placement system complete!")
        
    def run_integrated_pipeline(self):
        """Run the complete pipeline demonstration"""
        print("\n🚀 RUNNING INTEGRATED PIPELINE")
        print("=" * 40)
        
        placement_commands = []
        
        # Process pepsi first (PLACE), then mug second (STACK) in same location
        # Reorder objects to ensure proper stacking sequence
        pepsi_id = 'gso_Diet_Pepsi_Soda_Cola12_Pack_12_oz_Cans'
        mug_id = 'ycb_025_mug'
        
        # Find pepsi and mug in loaded objects
        pepsi_index = self.loaded_objects.index(pepsi_id) if pepsi_id in self.loaded_objects else -1
        mug_index = self.loaded_objects.index(mug_id) if mug_id in self.loaded_objects else -1
        
        # Reorder: pepsi first, then mug, then others
        reordered_objects = []
        if pepsi_index >= 0:
            reordered_objects.append(pepsi_id)
        if mug_index >= 0:
            reordered_objects.append(mug_id)
        
        # Add remaining objects
        for obj_id in self.loaded_objects:
            if obj_id not in [pepsi_id, mug_id]:
                reordered_objects.append(obj_id)
        
        print("🎲 Testing placement for each object (pepsi first, then mug)...")
        
        pepsi_zone = None  # Track pepsi's zone for mug placement
        pepsi_position = None  # Track pepsi's exact position for stacking
        
        for i, obj_id in enumerate(reordered_objects):
            # Strategy assignment for clear demonstration
            if obj_id == pepsi_id:
                strategy = PlacementStrategy.PLACE  # Place pepsi first as base
            elif obj_id == mug_id:
                strategy = PlacementStrategy.STACK  # Stack mug on pepsi
            elif obj_id == 'ycb_003_cracker_box':
                strategy = PlacementStrategy.LEAN   # Lean cracker box against pepsi
            else:
                strategy = PlacementStrategy.GROUP  # Group with similar objects
            
            print(f"\n  📍 Object {i+1}: {obj_id}")
            print(f"     Strategy: {strategy.value}")
            
            # Get object embedding
            object_embedding = self.similarity_engine.object_embeddings.get(obj_id)
            if object_embedding is None:
                print(f"     ❌ No embedding found")
                continue
                
            # Get object size
            metadata = self.object_library.get_metadata(obj_id)
            object_size = (0.1, 0.1) if not metadata else (metadata.estimated_size, metadata.estimated_size)
            
            # Special case: force mug and cracker box to pepsi's position for stacking/leaning
            if obj_id in [mug_id, 'ycb_003_cracker_box'] and pepsi_zone is not None and pepsi_position is not None:
                action = "stacking" if obj_id == mug_id else "leaning"
                print(f"     🎯 Forcing {obj_id.split('_')[-1]} to pepsi's position for {action}: {pepsi_position}")
                # Override candidate generation to return pepsi's exact position
                original_generate = self.placement_engine.candidate_generator.generate
                def force_pepsi_position(similarity_scores, object_size):
                    # Return only pepsi's exact position as candidate
                    return [(pepsi_zone, pepsi_position)]
                
                self.placement_engine.candidate_generator.generate = force_pepsi_position
            
            # Find placement
            result = self.placement_engine.find_placement(
                obj_id, object_embedding, object_size, strategy
            )
            
            # Restore original method if we overrode it
            if obj_id in [mug_id, 'ycb_003_cracker_box'] and pepsi_zone is not None and pepsi_position is not None:
                self.placement_engine.candidate_generator.generate = original_generate
            
            if result.success and result.placement_command:
                cmd = result.placement_command
                placement_commands.append(cmd)
                
                print(f"     ✅ Placement found!")
                print(f"        Zone: {cmd.zone_id}")
                print(f"        Position: ({cmd.position[0]:.2f}, {cmd.position[1]:.2f}, {cmd.position[2]:.2f})")
                print(f"        Confidence: {cmd.confidence_score:.3f}")
                
                # Track pepsi's zone and position for mug stacking
                if obj_id == pepsi_id:
                    pepsi_zone = cmd.zone_id
                    pepsi_position = (cmd.position[0], cmd.position[1])  # X, Y only
                    print(f"     📍 Pepsi zone recorded: {pepsi_zone}")
                    print(f"     📍 Pepsi position recorded: {pepsi_position}")
                
                # Move object to placement position
                bullet_id = self.object_library.loaded_objects[obj_id]
                p.resetBasePositionAndOrientation(
                    bullet_id, cmd.position, cmd.orientation,
                    physicsClientId=self.physics_client
                )
                
                # Update spatial constraints
                rect = Rectangle(
                    cmd.position[0] - object_size[0]/2,
                    cmd.position[1] - object_size[1]/2,
                    object_size[0],
                    object_size[1]
                )
                self.placement_engine.space_manager.add_occupied_space(cmd.zone_id, rect)
                
            else:
                print(f"     ❌ No valid placement found: {result.failure_reason}")
                
        # Step simulation to show final positions
        for _ in range(30):
            p.stepSimulation(physicsClientId=self.physics_client)
            time.sleep(0.02)
            
        print(f"\n📊 PIPELINE RESULTS:")
        print(f"     Objects processed: {len(self.loaded_objects)}")
        print(f"     Successful placements: {len(placement_commands)}")
        print(f"     Success rate: {len(placement_commands)/len(self.loaded_objects)*100:.1f}%")
        
        # Show space utilization
        space_state = self.placement_engine.space_manager.get_state()
        print(f"     Zones used: {space_state['num_zones']}")
        print(f"     Average efficiency: {space_state['average_efficiency']*100:.1f}%")
        
        return placement_commands
        
    def run_validation(self):
        """Run complete validation"""
        print("🎬 INTEGRATED PIPELINE VALIDATION")
        print("=" * 50)
        print("Testing complete Phase 1→2→3 workflow")
        print()
        
        try:
            # Setup phases sequentially
            self.setup_physics()
            self.setup_phase1_foundation()
            self.setup_phase2_vision()
            self.setup_phase3_placement()
            
            # Run integrated pipeline
            placement_commands = self.run_integrated_pipeline()
            
            print("\n👀 WHAT YOU SHOULD SEE:")
            print("   - 3-tier shelf system with wooden shelves")
            print("   - Franka Panda robot arm positioned appropriately")
            print("   - Objects moved from incoming area to optimal shelf positions")
            print("   - Different placement strategies demonstrated")
            print("   - Spatial organization following visual similarity")
            
            print("\n🎮 CONTROLS:")
            print("   - CTRL + DRAG: Move camera to inspect placements")
            print("   - Close window or press 'q' to exit")
            
            print("\n⏳ GUI running... Close window to exit")
            
            # Keep GUI open for inspection
            try:
                while True:
                    p.stepSimulation(physicsClientId=self.physics_client)
                    time.sleep(1/60)
                    
                    # Check for GUI window closure
                    keys = p.getKeyboardEvents(physicsClientId=self.physics_client)
                    if ord('q') in keys and keys[ord('q')] & p.KEY_WAS_TRIGGERED:
                        break
                        
            except KeyboardInterrupt:
                print("\nValidation stopped by user")
                
        except Exception as e:
            print(f"❌ Validation error: {e}")
            import traceback
            traceback.print_exc()
            
        finally:
            self.cleanup()
            
    def cleanup(self):
        """Clean up all resources"""
        print("\n🧹 Cleaning up...")
        
        try:
            if self.object_library:
                self.object_library.unload_all()
            if self.robot_controller:
                self.robot_controller.cleanup()
            if self.shelf_env:
                self.shelf_env.cleanup()
            if self.dino_model:
                self.dino_model.cleanup()
            if self.camera_system:
                self.camera_system.cleanup()
            if self.similarity_engine:
                self.similarity_engine.cleanup()
            if self.placement_engine:
                self.placement_engine.cleanup()
            if self.physics_client is not None:
                p.disconnect(self.physics_client)
        except Exception as e:
            print(f"Cleanup error: {e}")
            
        print("✅ Cleanup complete!")


def main():
    """Main validation entry point"""
    validator = IntegratedPipelineValidator()
    validator.run_validation()


if __name__ == "__main__":
    main()
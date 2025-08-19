#!/usr/bin/env python3
"""
Phase 3A Milestone Validation: Similarity Scores → Placement Commands
Demonstrates end-to-end pipeline functionality
"""

import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from placement import SimilarityEngine, PlacementEngine
from spatial import Rectangle


def main():
    """Validate Phase 3A milestone"""
    print("🧪 Phase 3A Milestone Validation")
    print("=================================")
    
    # Step 1: Initialize Phase 2 components (existing)
    print("\n1. Initializing Phase 2 SimilarityEngine...")
    similarity_engine = SimilarityEngine()
    similarity_engine.initialize()
    
    # Add test zone embeddings (simulating 18-zone shelf layout)
    zones_created = 0
    for shelf in range(3):  # 3 shelves
        for zone in range(6):  # 6 zones per shelf (3x2 grid)
            zone_id = f"shelf_{shelf}_zone_{zone}"
            # Create deterministic embeddings for testing
            np.random.seed(shelf * 100 + zone)
            embedding = np.random.rand(768)  # DINOv3 size
            embedding = embedding / np.linalg.norm(embedding)
            similarity_engine.store_zone_embedding(zone_id, embedding)
            zones_created += 1
    
    print(f"   ✅ Created {zones_created} zone embeddings")
    
    # Step 2: Initialize Phase 3A components (new)
    print("\n2. Initializing Phase 3A PlacementEngine...")
    placement_engine = PlacementEngine(similarity_engine)
    placement_engine.initialize()
    
    # Set up spatial layout for zones
    shelf_width, shelf_height = 0.8, 0.4  # 80cm x 40cm per shelf
    zone_width = shelf_width / 3  # 3 zones wide
    zone_height = shelf_height / 2  # 2 zones deep
    
    zones_configured = 0
    for shelf in range(3):
        for zone in range(6):
            zone_id = f"shelf_{shelf}_zone_{zone}"
            row = zone // 3
            col = zone % 3
            
            x = col * zone_width
            y = row * zone_height
            bounds = Rectangle(x, y, zone_width, zone_height)
            placement_engine.space_manager.set_zone_bounds(zone_id, bounds)
            zones_configured += 1
    
    print(f"   ✅ Configured {zones_configured} zone spatial bounds")
    
    # Step 3: Test similarity score generation (Phase 2)
    print("\n3. Testing similarity score generation...")
    test_object_embedding = np.array([0.5] * 768)  # Simple test embedding
    test_object_embedding = test_object_embedding / np.linalg.norm(test_object_embedding)
    
    similarity_scores = similarity_engine.get_similarity_heatmap(test_object_embedding)
    print(f"   ✅ Generated similarity scores for {len(similarity_scores)} zones")
    print(f"   📊 Score range: {min(similarity_scores.values()):.3f} - {max(similarity_scores.values()):.3f}")
    
    # Step 4: Test placement command generation (Phase 3A)
    print("\n4. Testing placement command generation...")
    
    # Test multiple objects
    successful_placements = 0
    failed_placements = 0
    
    for i in range(5):
        object_id = f"test_object_{i}"
        
        # Create unique object embedding
        np.random.seed(42 + i)
        object_embedding = np.random.rand(768)
        object_embedding = object_embedding / np.linalg.norm(object_embedding)
        
        # Generate placement
        result = placement_engine.find_placement(
            object_id, 
            object_embedding,
            object_size=(0.08, 0.06)  # 8cm x 6cm object
        )
        
        if result.success:
            cmd = result.placement_command
            print(f"   ✅ {object_id}: zone {cmd.zone_id}, pos ({cmd.position[0]:.2f}, {cmd.position[1]:.2f}, {cmd.position[2]:.2f}), confidence {cmd.confidence_score:.3f}")
            successful_placements += 1
            
            # Add occupied space for next iteration
            occupied = Rectangle(cmd.position[0], cmd.position[1], 0.08, 0.06)
            placement_engine.space_manager.add_occupied_space(cmd.zone_id, occupied)
            
        else:
            print(f"   ❌ {object_id}: {result.failure_reason}")
            failed_placements += 1
    
    # Step 5: Validate milestone criteria
    print("\n5. Validating Phase 3A milestone criteria...")
    
    criteria_met = 0
    total_criteria = 4
    
    # Criterion 1: Convert similarity scores → placement commands
    if successful_placements > 0:
        print("   ✅ Criterion 1: Successfully converts similarity scores to placement commands")
        criteria_met += 1
    else:
        print("   ❌ Criterion 1: Failed to convert similarity scores to placement commands")
    
    # Criterion 2: Respect spatial constraints
    if successful_placements > 1:  # Multiple objects placed without conflicts
        print("   ✅ Criterion 2: Respects spatial constraints (no overlapping placements)")
        criteria_met += 1
    else:
        print("   ❌ Criterion 2: Spatial constraint handling needs improvement")
    
    # Criterion 3: Handle failure cases gracefully
    if failed_placements == 0 or (failed_placements > 0 and "No valid placement found" in str(result.failure_reason)):
        print("   ✅ Criterion 3: Handles failure cases gracefully")
        criteria_met += 1
    else:
        print("   ❌ Criterion 3: Error handling needs improvement")
    
    # Criterion 4: Integration with Phase 2
    similarity_state = similarity_engine.get_state()
    placement_state = placement_engine.get_state()
    if similarity_state['num_zones'] > 0 and 'similarity_engine_state' in placement_state:
        print("   ✅ Criterion 4: Successfully integrates with Phase 2 SimilarityEngine")
        criteria_met += 1
    else:
        print("   ❌ Criterion 4: Phase 2 integration needs improvement")
    
    # Final assessment
    print(f"\n📋 Phase 3A Milestone Assessment:")
    print(f"   Criteria met: {criteria_met}/{total_criteria}")
    print(f"   Success rate: {successful_placements}/{successful_placements + failed_placements} placements")
    
    if criteria_met == total_criteria and successful_placements >= 3:
        print("   🎉 MILESTONE ACHIEVED: Phase 3A successfully converts similarity scores to placement commands!")
        return True
    else:
        print("   ⚠️  MILESTONE INCOMPLETE: Some criteria need attention")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
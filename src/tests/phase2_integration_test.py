#!/usr/bin/env python3
"""
Phase 2 Integration Test
Simple validation that vision system components work together
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from vision import ShelfZoneManager, FeatureCache, DinoModel
from placement import SimilarityEngine

def test_phase2_integration():
    """Test basic Phase 2 component integration"""
    print("🚀 PHASE 2 VISION SYSTEM - INTEGRATION TEST")
    print("=" * 50)
    
    # Test 1: Zone Manager
    print("📍 Testing ShelfZoneManager...")
    zone_manager = ShelfZoneManager()
    zone_manager.initialize()
    zones = zone_manager.get_all_zones()
    assert len(zones) == 18, f"Expected 18 zones, got {len(zones)}"
    print(f"✅ Created {len(zones)} zones")
    
    # Test 2: Feature Cache
    print("💾 Testing FeatureCache...")
    cache = FeatureCache()
    test_embedding = np.random.random(768)
    test_embedding /= np.linalg.norm(test_embedding)  # Normalize
    
    cache.store("test_object", test_embedding)
    loaded_embedding = cache.load("test_object")
    assert loaded_embedding is not None, "Failed to load cached embedding"
    np.testing.assert_array_almost_equal(test_embedding, loaded_embedding)
    print("✅ Cache store/load working")
    
    # Test 3: Similarity Engine
    print("🎯 Testing SimilarityEngine...")
    engine = SimilarityEngine()
    engine.initialize()
    
    # Add mock embeddings for all 18 zones
    for i in range(18):
        zone_embedding = np.random.random(768)
        zone_embedding /= np.linalg.norm(zone_embedding)
        engine.store_zone_embedding(f"zone_{i}", zone_embedding)
    
    # Test object similarity
    object_embedding = np.random.random(768)
    object_embedding /= np.linalg.norm(object_embedding)
    
    best_zone = engine.find_best_zone(object_embedding)
    heatmap = engine.get_similarity_heatmap(object_embedding)
    
    assert best_zone is not None, "No best zone found"
    assert len(heatmap) == 18, f"Expected 18 heatmap entries, got {len(heatmap)}"
    assert best_zone in heatmap, "Best zone not in heatmap"
    print(f"✅ Best placement: {best_zone}")
    print(f"✅ Similarity scores: {len(heatmap)} zones")
    
    # Test 4: End-to-end pipeline
    print("🔗 Testing end-to-end pipeline...")
    
    # Mock object embeddings
    object_embeddings = {}
    for i in range(5):  # 5 test objects
        obj_embedding = np.random.random(768)
        obj_embedding /= np.linalg.norm(obj_embedding)
        object_embeddings[f"object_{i}"] = obj_embedding
        engine.store_object_embedding(f"object_{i}", obj_embedding)
    
    # Place all objects
    placements = {}
    for obj_id, obj_embedding in object_embeddings.items():
        best_zone = engine.find_best_zone(obj_embedding)
        zone_center = zone_manager.get_zone_center(best_zone)
        placements[obj_id] = {
            "zone": best_zone,
            "position": zone_center
        }
        
        # Cache the embedding
        cache.store(obj_id, obj_embedding)
    
    assert len(placements) == 5, "Not all objects placed"
    print(f"✅ Placed {len(placements)} objects")
    
    # Verify cached embeddings
    for obj_id in object_embeddings.keys():
        cached = cache.load(obj_id)
        assert cached is not None, f"Failed to load cached embedding for {obj_id}"
    
    print("✅ All embeddings cached successfully")
    
    # Test 5: State validation
    print("📊 Testing component states...")
    zone_state = zone_manager.get_state()
    engine_state = engine.get_state()
    
    assert zone_state["num_zones"] == 18
    assert engine_state["num_objects"] == 5
    assert engine_state["num_zones"] == 18
    
    print(f"✅ Zone manager: {zone_state['num_zones']} zones")
    print(f"✅ Similarity engine: {engine_state['num_objects']} objects, {engine_state['num_zones']} zones")
    
    # Cleanup
    zone_manager.cleanup()
    engine.cleanup()
    cache.clear()
    
    print("\n🎉 PHASE 2 INTEGRATION TEST: PASSED")
    return True

if __name__ == '__main__':
    try:
        test_phase2_integration()
        print("\n✅ All Phase 2 components working correctly!")
    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        sys.exit(1)
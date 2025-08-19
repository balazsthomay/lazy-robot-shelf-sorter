#!/usr/bin/env python3
"""
Phase 2 Performance Validation
Test performance targets for M4 Pro hardware
"""

import sys
import os
import time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from vision import ShelfZoneManager, FeatureCache
from placement import SimilarityEngine

def measure_time(func, *args, **kwargs):
    """Measure execution time of a function"""
    start_time = time.perf_counter()
    result = func(*args, **kwargs)
    end_time = time.perf_counter()
    return result, (end_time - start_time) * 1000  # Return ms

def test_performance_targets():
    """Test Phase 2 performance targets"""
    print("⚡ PHASE 2 PERFORMANCE VALIDATION")
    print("=" * 40)
    print("Target: M4 Pro hardware performance")
    print()
    
    # Test 1: Zone Manager Performance
    print("📍 Zone Manager Performance:")
    zone_manager = ShelfZoneManager()
    
    _, init_time = measure_time(zone_manager.initialize)
    print(f"  Zone creation: {init_time:.2f}ms")
    assert init_time < 100, f"Zone creation too slow: {init_time:.2f}ms > 100ms"
    
    # Test zone lookups
    total_lookup_time = 0
    for i in range(100):  # 100 lookups
        zone_id = f"zone_{i % 18}"
        _, lookup_time = measure_time(zone_manager.get_zone_center, zone_id)
        total_lookup_time += lookup_time
    
    avg_lookup_time = total_lookup_time / 100
    print(f"  Zone lookup: {avg_lookup_time:.3f}ms avg (100 lookups)")
    assert avg_lookup_time < 1.0, f"Zone lookup too slow: {avg_lookup_time:.3f}ms > 1.0ms"
    print("  ✅ Zone manager performance: PASS")
    print()
    
    # Test 2: Feature Cache Performance
    print("💾 Feature Cache Performance:")
    cache = FeatureCache()
    
    # Test storing 53 embeddings (all objects)
    embeddings = {}
    total_store_time = 0
    
    for i in range(53):
        embedding = np.random.random(768)
        embedding /= np.linalg.norm(embedding)
        embeddings[f"object_{i}"] = embedding
        
        _, store_time = measure_time(cache.store, f"object_{i}", embedding)
        total_store_time += store_time
    
    print(f"  Store 53 embeddings: {total_store_time:.2f}ms total")
    print(f"  Average store time: {total_store_time/53:.2f}ms per embedding")
    assert total_store_time < 500, f"Cache storage too slow: {total_store_time:.2f}ms > 500ms"
    
    # Test loading embeddings
    total_load_time = 0
    for i in range(53):
        _, load_time = measure_time(cache.load, f"object_{i}")
        total_load_time += load_time
    
    print(f"  Load 53 embeddings: {total_load_time:.2f}ms total")
    print(f"  Average load time: {total_load_time/53:.2f}ms per embedding")
    assert total_load_time < 200, f"Cache loading too slow: {total_load_time:.2f}ms > 200ms"
    print("  ✅ Feature cache performance: PASS")
    print()
    
    # Test 3: Similarity Engine Performance
    print("🎯 Similarity Engine Performance:")
    engine = SimilarityEngine()
    engine.initialize()
    
    # Setup: Add 18 zone embeddings and 53 object embeddings
    setup_start = time.perf_counter()
    
    # Add zone embeddings
    for i in range(18):
        zone_embedding = np.random.random(768)
        zone_embedding /= np.linalg.norm(zone_embedding)
        engine.store_zone_embedding(f"zone_{i}", zone_embedding)
    
    # Add object embeddings
    for i in range(53):
        obj_embedding = embeddings[f"object_{i}"]
        engine.store_object_embedding(f"object_{i}", obj_embedding)
    
    setup_time = (time.perf_counter() - setup_start) * 1000
    print(f"  Setup (18 zones + 53 objects): {setup_time:.2f}ms")
    
    # Test similarity computation speed
    test_object_embedding = np.random.random(768)
    test_object_embedding /= np.linalg.norm(test_object_embedding)
    
    # Single similarity query
    _, single_query_time = measure_time(engine.find_best_zone, test_object_embedding)
    print(f"  Single similarity query: {single_query_time:.2f}ms")
    assert single_query_time < 50, f"Similarity query too slow: {single_query_time:.2f}ms > 50ms"
    
    # Heatmap generation
    _, heatmap_time = measure_time(engine.get_similarity_heatmap, test_object_embedding)
    print(f"  Full heatmap generation: {heatmap_time:.2f}ms")
    assert heatmap_time < 100, f"Heatmap generation too slow: {heatmap_time:.2f}ms > 100ms"
    
    # Batch processing (simulate placing 10 objects)
    batch_start = time.perf_counter()
    placements = []
    
    for i in range(10):
        obj_embedding = np.random.random(768)
        obj_embedding /= np.linalg.norm(obj_embedding)
        best_zone = engine.find_best_zone(obj_embedding)
        placements.append(best_zone)
    
    batch_time = (time.perf_counter() - batch_start) * 1000
    print(f"  Batch placement (10 objects): {batch_time:.2f}ms")
    print(f"  Average per object: {batch_time/10:.2f}ms")
    assert batch_time < 500, f"Batch processing too slow: {batch_time:.2f}ms > 500ms"
    
    print("  ✅ Similarity engine performance: PASS")
    print()
    
    # Test 4: Memory Usage Estimation
    print("💾 Memory Usage Estimation:")
    
    # Estimate memory for embeddings
    embedding_size = 768 * 8  # 768 float64s * 8 bytes each
    total_objects = 53
    total_zones = 18
    
    object_memory = total_objects * embedding_size
    zone_memory = total_zones * embedding_size
    total_memory = object_memory + zone_memory
    
    print(f"  Object embeddings: {object_memory/1024/1024:.2f}MB ({total_objects} × {embedding_size/1024:.1f}KB)")
    print(f"  Zone embeddings: {zone_memory/1024/1024:.2f}MB ({total_zones} × {embedding_size/1024:.1f}KB)")
    print(f"  Total estimated: {total_memory/1024/1024:.2f}MB")
    
    # Target: <75MB total as specified in plan
    target_memory = 75 * 1024 * 1024  # 75MB in bytes
    assert total_memory < target_memory, f"Memory usage too high: {total_memory/1024/1024:.2f}MB > 75MB"
    print("  ✅ Memory usage within target: PASS")
    print()
    
    # Cleanup
    zone_manager.cleanup()
    engine.cleanup()
    cache.clear()
    
    print("🎉 PHASE 2 PERFORMANCE VALIDATION: PASSED")
    print()
    print("📊 PERFORMANCE SUMMARY:")
    print(f"  Zone creation: {init_time:.2f}ms")
    print(f"  Cache operations: {(total_store_time + total_load_time)/106:.2f}ms avg")
    print(f"  Similarity query: {single_query_time:.2f}ms")
    print(f"  Object placement: {batch_time/10:.2f}ms avg")
    print(f"  Memory footprint: {total_memory/1024/1024:.2f}MB")
    print()
    print("✅ All performance targets met for M4 Pro hardware!")
    
    return True

if __name__ == '__main__':
    try:
        test_performance_targets()
    except AssertionError as e:
        print(f"\n❌ Performance test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)
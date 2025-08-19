#!/usr/bin/env python3
"""
Real DINO Model Test
Test actual HuggingFace model loading and feature extraction
"""

import sys
import os
import numpy as np
from PIL import Image
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

def test_real_dino_loading():
    """Test loading actual DINO models from HuggingFace"""
    print("🤖 REAL DINO MODEL TEST")
    print("=" * 40)
    
    try:
        from vision import DinoModel
        print("✅ Vision module imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import vision module: {e}")
        return False
    
    # Test 1: Check dependencies
    print("\n📋 Checking dependencies...")
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
    except ImportError:
        print("❌ PyTorch not available - install with: pip install torch")
        return False
        
    try:
        import transformers
        print(f"✅ Transformers: {transformers.__version__}")
    except ImportError:
        print("❌ Transformers not available - install with: pip install transformers")
        return False
    
    # Test 2: Initialize DINO model
    print("\n🔧 Testing DINO model initialization...")
    try:
        dino = DinoModel()
        print("✅ DinoModel created")
        
        start_time = time.time()
        dino.initialize()
        load_time = time.time() - start_time
        
        print(f"✅ Model loaded in {load_time:.2f} seconds")
        print(f"✅ Model info: {dino.get_model_info()}")
        
    except Exception as e:
        print(f"❌ Failed to initialize DINO model: {e}")
        print("\n🔍 Troubleshooting tips:")
        print("1. Ensure you're authenticated with HuggingFace")
        print("2. Check internet connection")
        print("3. Verify HuggingFace Hub access")
        return False
    
    # Test 3: Create test image
    print("\n🖼️ Testing feature extraction...")
    try:
        # Create a simple test image
        test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        test_pil_image = Image.fromarray(test_image)
        
        print("✅ Test image created (224x224x3)")
        
        # Extract features
        start_time = time.time()
        features = dino.extract_features(test_pil_image)
        extraction_time = time.time() - start_time
        
        print(f"✅ Features extracted in {extraction_time*1000:.2f}ms")
        print(f"✅ Feature shape: {features.shape}")
        print(f"✅ Feature norm: {np.linalg.norm(features):.6f} (should be ~1.0)")
        
        # Verify features are normalized
        assert abs(np.linalg.norm(features) - 1.0) < 0.01, "Features not properly normalized"
        print("✅ Features properly normalized")
        
    except Exception as e:
        print(f"❌ Feature extraction failed: {e}")
        return False
    
    # Test 4: Test with multiple images
    print("\n🔄 Testing consistency with multiple images...")
    try:
        embeddings = []
        for i in range(3):
            test_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            embedding = dino.extract_features(Image.fromarray(test_img))
            embeddings.append(embedding)
        
        # Check all embeddings are normalized
        for i, emb in enumerate(embeddings):
            norm = np.linalg.norm(emb)
            assert abs(norm - 1.0) < 0.01, f"Embedding {i} not normalized: {norm}"
        
        print("✅ All embeddings properly normalized")
        print("✅ Consistent feature extraction working")
        
    except Exception as e:
        print(f"❌ Consistency test failed: {e}")
        return False
    
    # Test 5: Cleanup
    print("\n🧹 Testing cleanup...")
    try:
        dino.cleanup()
        print("✅ Model cleanup successful")
    except Exception as e:
        print(f"❌ Cleanup failed: {e}")
        return False
    
    print("\n🎉 ALL DINO MODEL TESTS PASSED!")
    print(f"📊 Performance: {extraction_time*1000:.1f}ms per image")
    print("🚀 Ready for full vision system integration!")
    return True

def test_end_to_end_vision_pipeline():
    """Test complete vision pipeline with real models"""
    print("\n\n🔗 END-TO-END VISION PIPELINE TEST")
    print("=" * 50)
    
    try:
        from vision import DinoModel, ShelfZoneManager, FeatureCache
        from placement import SimilarityEngine
        
        # Initialize all components
        print("🔧 Initializing components...")
        dino = DinoModel()
        dino.initialize()
        
        zones = ShelfZoneManager()
        zones.initialize()
        
        engine = SimilarityEngine()
        engine.initialize()
        
        cache = FeatureCache()
        
        print("✅ All components initialized")
        
        # Simulate zone feature extraction
        print("📍 Extracting zone features...")
        zone_count = 0
        for zone_id, zone_center in zones.get_all_zones().items():
            # Create mock zone image (in real usage, this comes from camera)
            zone_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            zone_features = dino.extract_features(Image.fromarray(zone_image))
            
            # Store in similarity engine and cache
            engine.store_zone_embedding(zone_id, zone_features)
            cache.store(f"zone_{zone_id}", zone_features)
            zone_count += 1
        
        print(f"✅ Extracted features for {zone_count} zones")
        
        # Test object placement
        print("📦 Testing object placement...")
        placement_results = []
        
        for i in range(5):  # Test 5 objects
            # Create mock object image
            object_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            object_features = dino.extract_features(Image.fromarray(object_image))
            
            # Find best placement
            best_zone = engine.find_best_zone(object_features)
            heatmap = engine.get_similarity_heatmap(object_features)
            zone_position = zones.get_zone_center(best_zone)
            
            placement_results.append({
                'object_id': f'object_{i}',
                'best_zone': best_zone,
                'position': zone_position,
                'confidence': heatmap[best_zone]
            })
            
            # Cache object features
            cache.store(f"object_{i}", object_features)
        
        print(f"✅ Placed {len(placement_results)} objects")
        
        # Show results
        print("\n📊 PLACEMENT RESULTS:")
        for result in placement_results:
            print(f"  {result['object_id']} → {result['best_zone']} "
                  f"(confidence: {result['confidence']:.3f})")
        
        # Cleanup
        dino.cleanup()
        zones.cleanup() 
        engine.cleanup()
        cache.clear()
        
        print("\n🎉 END-TO-END PIPELINE SUCCESS!")
        return True
        
    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    print("🧪 COMPREHENSIVE DINO MODEL TESTING")
    print("=" * 50)
    
    success = test_real_dino_loading()
    
    if success:
        success = test_end_to_end_vision_pipeline()
    
    if success:
        print("\n✅ ALL TESTS PASSED - VISION SYSTEM READY!")
    else:
        print("\n❌ SOME TESTS FAILED - SEE ABOVE FOR DETAILS")
        sys.exit(1)
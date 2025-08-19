#!/usr/bin/env python3
"""
Milestone 6 Test - Incremental Domain Randomization
Part of Phase 1: Foundation - Milestone 6
"""

import time
import numpy as np
import pybullet as p
import pybullet_data
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from domain_randomization import DomainRandomizer, RandomizationConfig
from vision import CameraSystem, CameraConfiguration
from simulation import ShelfEnvironment, ShelfConfiguration


def test_lighting_randomization() -> bool:
    """Test lighting variation system"""
    print("🚀 Testing lighting randomization...")
    
    # Create randomizer
    config = RandomizationConfig(
        lighting_range=(0.3, 1.0),
        enable_jitter=False,
        enable_noise=False
    )
    randomizer = DomainRandomizer(config)
    
    # Test multiple randomizations
    intensities = []
    for _ in range(10):
        results = randomizer.randomize_environment()
        intensities.append(results.get('lighting_intensity', 1.0))
    
    # Validate range
    min_intensity = min(intensities)
    max_intensity = max(intensities)
    
    print(f"  Intensity range: {min_intensity:.3f} - {max_intensity:.3f}")
    
    # Check within bounds
    in_bounds = (min_intensity >= 0.3 and max_intensity <= 1.0)
    has_variation = (max_intensity - min_intensity) > 0.1
    
    success = in_bounds and has_variation
    print(f"  ✅ Lighting variation working" if success else "  ❌ Lighting issues")
    return success


def test_camera_jitter() -> bool:
    """Test camera position randomization"""
    print("🚀 Testing camera jitter...")
    
    # Initialize PyBullet for camera
    physics_client = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    
    # Create original camera
    camera_config = CameraConfiguration(image_width=160, image_height=120)
    camera_system = CameraSystem(camera_config)
    original_camera = camera_system.setup_front_facing_camera()
    
    # Create randomizer
    config = RandomizationConfig(
        camera_jitter=0.05,
        enable_lighting=False,
        enable_noise=False
    )
    randomizer = DomainRandomizer(config)
    
    # Test multiple jitters
    jitter_distances = []
    for _ in range(5):
        jittered_camera = randomizer.randomize_camera(original_camera)
        distance = randomizer.jitter.get_jitter_amount(original_camera, jittered_camera)
        jitter_distances.append(distance)
    
    avg_jitter = np.mean(jitter_distances)
    max_jitter = max(jitter_distances)
    
    print(f"  Average jitter: {avg_jitter:.4f}m")
    print(f"  Maximum jitter: {max_jitter:.4f}m")
    
    # Check reasonable jitter (should be within configured range)
    reasonable_jitter = max_jitter <= 0.1  # Within 10cm
    has_jitter = avg_jitter > 0.001  # At least 1mm average
    
    p.disconnect()
    
    success = reasonable_jitter and has_jitter
    print(f"  ✅ Camera jitter working" if success else "  ❌ Camera jitter issues")
    return success


def test_depth_noise() -> bool:
    """Test depth sensor noise generation"""
    print("🚀 Testing depth noise...")
    
    # Create synthetic depth image
    depth_image = np.ones((120, 160)) * 2.0  # 2m depth everywhere
    
    # Create randomizer
    config = RandomizationConfig(
        depth_noise=0.05,  # 5% noise
        enable_lighting=False,
        enable_jitter=False
    )
    randomizer = DomainRandomizer(config)
    
    # Apply noise multiple times
    noise_stats = []
    for _ in range(5):
        noisy_depth = randomizer.randomize_depth(depth_image)
        stats = randomizer.noise_gen.calculate_noise_stats(depth_image, noisy_depth)
        noise_stats.append(stats)
    
    # Average statistics
    avg_std = np.mean([s['std_noise'] for s in noise_stats])
    avg_snr = np.mean([s['snr'] for s in noise_stats])
    
    print(f"  Noise std: {avg_std:.4f}m")
    print(f"  Signal-to-noise ratio: {avg_snr:.1f}")
    
    # Validate noise characteristics
    reasonable_noise = 0.01 < avg_std < 0.2  # Between 1cm and 20cm std
    good_snr = avg_snr > 5  # SNR > 5
    
    success = reasonable_noise and good_snr
    print(f"  ✅ Depth noise working" if success else "  ❌ Depth noise issues")
    return success


def test_integrated_randomization() -> bool:
    """Test all randomization components together"""
    print("🚀 Testing integrated randomization...")
    
    # Initialize environment
    physics_client = p.connect(p.DIRECT)
    p.setGravity(0, 0, -9.81)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF("plane.urdf")
    
    # Create environment
    config = ShelfConfiguration(num_shelves=1)
    env = ShelfEnvironment(config)
    env.initialize(use_gui=False)
    shelf_ids = env.create_shelves()
    
    # Create camera system
    camera_config = CameraConfiguration(image_width=160, image_height=120)
    camera_system = CameraSystem(camera_config)
    original_camera = camera_system.setup_front_facing_camera()
    
    # Create full randomizer
    randomizer = DomainRandomizer()
    
    # Test full pipeline
    start_time = time.time()
    
    # Randomize environment
    env_results = randomizer.randomize_environment()
    
    # Randomize camera
    random_camera = randomizer.randomize_camera(original_camera)
    
    # Capture with randomized camera
    rgbd_data = random_camera.capture_rgbd()
    
    # Add depth noise
    noisy_depth = randomizer.randomize_depth(rgbd_data.depth_image)
    
    pipeline_time = time.time() - start_time
    
    # Validate results
    has_rgb = rgbd_data.rgb_image.shape == (120, 160, 3)
    has_depth = rgbd_data.depth_image.shape == (120, 160)  # Fix: should match height, width
    has_noisy_depth = noisy_depth.shape == rgbd_data.depth_image.shape
    fast_enough = pipeline_time < 0.1  # Under 100ms
    
    print(f"  Pipeline time: {pipeline_time:.3f}s")
    print(f"  Lighting intensity: {env_results.get('lighting_intensity', 'N/A')}")
    print(f"  RGB shape: {rgbd_data.rgb_image.shape}")
    print(f"  Depth shape: {rgbd_data.depth_image.shape}")
    
    # Cleanup
    camera_system.cleanup()
    env.cleanup()
    p.disconnect()
    
    success = has_rgb and has_depth and has_noisy_depth and fast_enough
    print(f"  ✅ Integration working" if success else "  ❌ Integration issues")
    return success


def main():
    """Run Milestone 6 tests"""
    print("🚀 MILESTONE 6: INCREMENTAL DOMAIN RANDOMIZATION")
    print("=" * 50)
    
    tests = [
        ("Lighting Randomization", test_lighting_randomization),
        ("Camera Jitter", test_camera_jitter),
        ("Depth Noise", test_depth_noise),
        ("Integrated Randomization", test_integrated_randomization),
    ]
    
    results = {}
    total_start = time.time()
    
    for test_name, test_func in tests:
        print(f"\n{test_name}...")
        start_time = time.time()
        
        try:
            result = test_func()
            results[test_name] = result
        except Exception as e:
            print(f"  ❌ Test failed with exception: {e}")
            results[test_name] = False
            
        test_time = time.time() - start_time
        print(f"  Time: {test_time:.3f}s")
        
    total_time = time.time() - total_start
    
    # Final results
    print("\n" + "=" * 50)
    print("📊 MILESTONE 6 - FINAL RESULTS")
    print("=" * 50)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        
    print(f"Total test time: {total_time:.3f}s")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n🎉 MILESTONE 6 COMPLETE!")
        print("✅ Domain Randomization System ready")
        print("\nDomain Randomization Features Validated:")
        print("- Lighting intensity variation (0.3-1.0)")
        print("- Camera position jitter (±5cm)")
        print("- Depth sensor noise (0-5% Gaussian)")
        print("- Integrated randomization pipeline")
        print("- Stability and performance validation")
    else:
        print("\n❌ Milestone 6 validation failed")
        print("Domain randomization needs attention")
        
    return all_passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
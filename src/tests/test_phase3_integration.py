#!/usr/bin/env python3
"""
Phase 3D Integration Tests
Comprehensive end-to-end testing for Phase 2→3→4 boundary management
"""

import unittest
import time
import numpy as np
from unittest.mock import Mock, patch
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from integration import (
    IntegratedPlacementEngine, FallbackManager, PhaseBoundaryManager,
    PlacementMetrics, FallbackReason, FallbackEntry
)
from placement import SimilarityEngine, PlacementStrategy, PlacementResult
from spatial import Rectangle


class TestFallbackManager(unittest.TestCase):
    """Test fallback management with time budgets and blacklists"""
    
    def setUp(self):
        self.fallback_manager = FallbackManager(time_budget_seconds=5.0)
    
    def test_blacklist_management(self):
        """Test blacklist add/check/cleanup functionality"""
        zone_id = "test_zone"
        strategy = PlacementStrategy.STACK
        
        # Initially not blacklisted
        self.assertFalse(self.fallback_manager.is_blacklisted(zone_id, strategy))
        
        # Add to blacklist
        self.fallback_manager.add_to_blacklist(zone_id, strategy, FallbackReason.STABILITY_FAILURE)
        
        # Should now be blacklisted
        self.assertTrue(self.fallback_manager.is_blacklisted(zone_id, strategy))
        
        # Different strategy should not be blacklisted
        self.assertFalse(self.fallback_manager.is_blacklisted(zone_id, PlacementStrategy.PLACE))
    
    def test_blacklist_expiration(self):
        """Test blacklist entries expire after penalty decay time"""
        zone_id = "test_zone"
        strategy = PlacementStrategy.STACK
        
        # Add entry with short decay time
        entry = FallbackEntry(
            zone_id=zone_id,
            strategy=strategy,
            failure_reason=FallbackReason.TIMEOUT,
            timestamp=time.time() - 400,  # 400 seconds ago
            penalty_decay_time=300.0      # 300 second decay
        )
        self.fallback_manager.blacklist.append(entry)
        
        # Should be expired and cleaned up
        self.assertFalse(self.fallback_manager.is_blacklisted(zone_id, strategy))
        self.assertEqual(len(self.fallback_manager.blacklist), 0)
    
    def test_time_budget_management(self):
        """Test time budget checking"""
        start_time = time.time()
        
        # Should initially allow continuation
        self.assertTrue(self.fallback_manager.should_continue(start_time))
        
        # Mock time passing beyond budget
        with patch('time.time', return_value=start_time + 6.0):  # 6 seconds > 5 second budget
            self.assertFalse(self.fallback_manager.should_continue(start_time))
    
    def test_fallback_strategy_ordering(self):
        """Test fallback strategy generation maintains order"""
        preferred = PlacementStrategy.LEAN
        strategies = self.fallback_manager.get_fallback_strategies(preferred)
        
        # Preferred strategy should be first
        self.assertEqual(strategies[0], preferred)
        
        # Should contain all strategies
        self.assertEqual(len(strategies), 4)
        self.assertIn(PlacementStrategy.PLACE, strategies)
        self.assertIn(PlacementStrategy.STACK, strategies)
        self.assertIn(PlacementStrategy.GROUP, strategies)


class TestIntegratedPlacementEngine(unittest.TestCase):
    """Test production-ready placement engine with fallback"""
    
    def setUp(self):
        """Set up test environment"""
        self.similarity_engine = SimilarityEngine()
        self.similarity_engine.initialize()
        
        # Add test zone embeddings
        for i in range(3):
            zone_id = f"shelf_0_zone_{i}"
            embedding = np.random.rand(768)
            embedding = embedding / np.linalg.norm(embedding)
            self.similarity_engine.store_zone_embedding(zone_id, embedding)
        
        self.integrated_engine = IntegratedPlacementEngine(
            self.similarity_engine, time_budget_seconds=10.0
        )
        
        # Set up spatial bounds
        for i in range(3):
            zone_id = f"shelf_0_zone_{i}"
            bounds = Rectangle(i * 1.0, 0.0, 1.0, 1.0)
            self.integrated_engine.placement_engine.space_manager.set_zone_bounds(zone_id, bounds)
    
    def test_successful_placement_with_fallback(self):
        """Test successful placement records metrics"""
        object_embedding = np.random.rand(768)
        object_embedding = object_embedding / np.linalg.norm(object_embedding)
        
        result = self.integrated_engine.find_placement_with_fallback(
            "test_object", object_embedding, (0.1, 0.1), PlacementStrategy.PLACE
        )
        
        self.assertTrue(result.success)
        self.assertIsNotNone(result.placement_command)
        
        # Check metrics were recorded
        metrics = self.integrated_engine.get_performance_metrics(last_n=1)
        self.assertEqual(metrics["placement_count"], 1)
        self.assertGreater(metrics["average_time_ms"], 0)
        self.assertEqual(metrics["success_rate"], 1.0)
    
    def test_fallback_strategy_progression(self):
        """Test engine tries multiple strategies on failure"""
        # Create engine with limited zones to force failures
        limited_engine = SimilarityEngine()
        limited_engine.initialize()
        
        # Add only one zone embedding
        zone_id = "shelf_0_zone_0"
        embedding = np.random.rand(768)
        embedding = embedding / np.linalg.norm(embedding)
        limited_engine.store_zone_embedding(zone_id, embedding)
        
        integrated_engine = IntegratedPlacementEngine(limited_engine, time_budget_seconds=1.0)
        
        # Set very small zone bounds to force constraint failures
        bounds = Rectangle(0.0, 0.0, 0.05, 0.05)  # Too small for 0.1x0.1 object
        integrated_engine.placement_engine.space_manager.set_zone_bounds(zone_id, bounds)
        
        object_embedding = np.random.rand(768)
        object_embedding = object_embedding / np.linalg.norm(object_embedding)
        
        result = integrated_engine.find_placement_with_fallback(
            "test_object", object_embedding, (0.1, 0.1), PlacementStrategy.STACK
        )
        
        # Should eventually fail but try multiple strategies
        metrics = integrated_engine.get_performance_metrics(last_n=1)
        self.assertGreaterEqual(metrics["average_fallback_steps"], 1)
    
    def test_timeout_handling(self):
        """Test placement respects time budget"""
        # Create engine with very short time budget
        short_budget_engine = IntegratedPlacementEngine(
            self.similarity_engine, time_budget_seconds=0.001  # 1ms budget
        )
        
        object_embedding = np.random.rand(768)
        object_embedding = object_embedding / np.linalg.norm(object_embedding)
        
        start_time = time.time()
        result = short_budget_engine.find_placement_with_fallback(
            "test_object", object_embedding, (0.1, 0.1)
        )
        elapsed_time = time.time() - start_time
        
        # Should complete quickly due to timeout
        self.assertLess(elapsed_time, 1.0)  # Should not take more than 1 second
    
    def test_blacklist_integration(self):
        """Test blacklist prevents retrying failed zone-strategy combinations"""
        # Manually add blacklist entry
        self.integrated_engine.fallback_manager.add_to_blacklist(
            "shelf_0_zone_0", PlacementStrategy.PLACE, FallbackReason.STABILITY_FAILURE
        )
        
        object_embedding = np.random.rand(768)
        object_embedding = object_embedding / np.linalg.norm(object_embedding)
        
        result = self.integrated_engine.find_placement_with_fallback(
            "test_object", object_embedding, (0.1, 0.1), PlacementStrategy.PLACE
        )
        
        # Should still succeed using other zones or strategies
        if result.success:
            # Should not use blacklisted zone-strategy combination
            self.assertNotEqual(result.placement_command.zone_id, "shelf_0_zone_0")
    
    def test_performance_metrics_collection(self):
        """Test comprehensive metrics collection"""
        # Perform several placements
        for i in range(3):
            object_embedding = np.random.rand(768)
            object_embedding = object_embedding / np.linalg.norm(object_embedding)
            
            self.integrated_engine.find_placement_with_fallback(
                f"test_object_{i}", object_embedding, (0.1, 0.1)
            )
        
        metrics = self.integrated_engine.get_performance_metrics(last_n=3)
        
        # Validate metrics structure
        self.assertEqual(metrics["placement_count"], 3)
        self.assertIn("average_time_ms", metrics)
        self.assertIn("success_rate", metrics)
        self.assertIn("average_fallback_steps", metrics)
        self.assertIn("performance_percentiles", metrics)
        self.assertIn("p50_ms", metrics["performance_percentiles"])
        self.assertIn("p90_ms", metrics["performance_percentiles"])


class TestPhaseBoundaryManager(unittest.TestCase):
    """Test cross-phase integration and error handling"""
    
    def setUp(self):
        """Set up cross-phase test environment"""
        self.similarity_engine = SimilarityEngine()
        self.similarity_engine.initialize()
        
        # Add test zones
        for i in range(3):
            zone_id = f"shelf_0_zone_{i}"
            embedding = np.random.rand(768)
            embedding = embedding / np.linalg.norm(embedding)
            self.similarity_engine.store_zone_embedding(zone_id, embedding)
        
        self.integrated_engine = IntegratedPlacementEngine(self.similarity_engine)
        
        # Set up spatial bounds
        for i in range(3):
            zone_id = f"shelf_0_zone_{i}"
            bounds = Rectangle(i * 1.0, 0.0, 1.0, 1.0)
            self.integrated_engine.placement_engine.space_manager.set_zone_bounds(zone_id, bounds)
        
        self.boundary_manager = PhaseBoundaryManager(
            self.similarity_engine, self.integrated_engine
        )
    
    def test_complete_pipeline_with_embedding(self):
        """Test Phase 2→3→4 pipeline with pre-computed embedding"""
        object_data = {
            'embedding': np.random.rand(768) / np.linalg.norm(np.random.rand(768)),
            'size': (0.1, 0.1),
            'strategy': PlacementStrategy.PLACE
        }
        
        result = self.boundary_manager.process_object_placement("test_object", object_data)
        
        self.assertTrue(result.success)
        self.assertIsNotNone(result.placement_command)
        self.assertGreater(result.placement_command.world_state_version, 0)
    
    def test_pipeline_fallback_feature_extraction(self):
        """Test pipeline falls back to feature extraction when no embedding provided"""
        object_data = {
            'image': np.random.rand(224, 224, 3),  # Mock image data
            'size': (0.1, 0.1)
        }
        
        result = self.boundary_manager.process_object_placement("test_object", object_data)
        
        # Should succeed using fallback feature extraction
        self.assertTrue(result.success)
    
    def test_world_state_versioning(self):
        """Test world state version increments correctly"""
        initial_version = self.boundary_manager.world_state_version
        
        object_data = {
            'embedding': np.random.rand(768) / np.linalg.norm(np.random.rand(768)),
            'size': (0.1, 0.1)
        }
        
        # Process multiple objects
        for i in range(3):
            result = self.boundary_manager.process_object_placement(f"object_{i}", object_data)
            if result.success:
                self.assertGreater(
                    result.placement_command.world_state_version, 
                    initial_version + i
                )
    
    def test_cross_phase_metrics(self):
        """Test comprehensive cross-phase metrics collection"""
        object_data = {
            'embedding': np.random.rand(768) / np.linalg.norm(np.random.rand(768)),
            'size': (0.1, 0.1)
        }
        
        self.boundary_manager.process_object_placement("test_object", object_data)
        
        metrics = self.boundary_manager.get_cross_phase_metrics()
        
        self.assertIn("world_state_version", metrics)
        self.assertIn("phase_2_state", metrics)
        self.assertIn("phase_3_metrics", metrics)
        self.assertGreater(metrics["world_state_version"], 0)
    
    def test_error_handling_and_recovery(self):
        """Test graceful error handling across phase boundaries"""
        # Test with malformed object data
        malformed_data = {
            'invalid_field': 'invalid_value'
        }
        
        result = self.boundary_manager.process_object_placement("test_object", malformed_data)
        
        # Should fail gracefully with meaningful error message
        self.assertFalse(result.success)
        self.assertIn("error", result.failure_reason.lower())


class TestPerformanceRequirements(unittest.TestCase):
    """Test Phase 3D performance requirements"""
    
    def setUp(self):
        """Set up performance test environment"""
        self.similarity_engine = SimilarityEngine()
        self.similarity_engine.initialize()
        
        # Add realistic number of zones (18 zones = 3 shelves × 6 zones)
        for shelf in range(3):
            for zone in range(6):
                zone_id = f"shelf_{shelf}_zone_{zone}"
                embedding = np.random.rand(768)
                embedding = embedding / np.linalg.norm(embedding)
                self.similarity_engine.store_zone_embedding(zone_id, embedding)
        
        self.integrated_engine = IntegratedPlacementEngine(
            self.similarity_engine, time_budget_seconds=30.0
        )
        
        # Set up all spatial bounds
        for shelf in range(3):
            for zone in range(6):
                zone_id = f"shelf_{shelf}_zone_{zone}"
                row = zone // 3
                col = zone % 3
                bounds = Rectangle(col * 0.27, row * 0.2, 0.27, 0.2)
                self.integrated_engine.placement_engine.space_manager.set_zone_bounds(zone_id, bounds)
    
    def test_thirty_second_performance_target(self):
        """Test <30s per object performance target (Phase 3D Milestone Gate)"""
        test_objects = []
        
        # Test with 5 different objects
        for i in range(5):
            object_embedding = np.random.rand(768)
            object_embedding = object_embedding / np.linalg.norm(object_embedding)
            object_id = f"performance_test_object_{i}"
            
            start_time = time.time()
            result = self.integrated_engine.find_placement_with_fallback(
                object_id, object_embedding, (0.08, 0.06)
            )
            elapsed_time = time.time() - start_time
            
            test_objects.append({
                'object_id': object_id,
                'elapsed_time': elapsed_time,
                'success': result.success
            })
        
        # Validate performance targets
        average_time = np.mean([obj['elapsed_time'] for obj in test_objects])
        max_time = max([obj['elapsed_time'] for obj in test_objects])
        success_rate = np.mean([obj['success'] for obj in test_objects])
        
        # Performance assertions
        self.assertLess(average_time, 30.0, f"Average time {average_time:.2f}s exceeds 30s target")
        self.assertLess(max_time, 30.0, f"Max time {max_time:.2f}s exceeds 30s target") 
        self.assertGreaterEqual(success_rate, 0.8, f"Success rate {success_rate:.2f} below 80%")
        
        print(f"\n🎯 Performance Test Results:")
        print(f"   Average time: {average_time:.2f}s (target: <30s)")
        print(f"   Max time: {max_time:.2f}s (target: <30s)")
        print(f"   Success rate: {success_rate:.2%} (target: ≥80%)")
    
    def test_memory_usage_tracking(self):
        """Test memory usage stays within reasonable bounds"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Process multiple objects
        for i in range(10):
            object_embedding = np.random.rand(768)
            object_embedding = object_embedding / np.linalg.norm(object_embedding)
            
            self.integrated_engine.find_placement_with_fallback(
                f"memory_test_object_{i}", object_embedding, (0.1, 0.1)
            )
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory should not increase significantly (target: <100MB additional from plan)
        self.assertLess(memory_increase, 100.0, 
                       f"Memory increased by {memory_increase:.1f}MB, exceeds 100MB target")
        
        print(f"\n💾 Memory Usage Test:")
        print(f"   Initial: {initial_memory:.1f}MB")
        print(f"   Final: {final_memory:.1f}MB") 
        print(f"   Increase: {memory_increase:.1f}MB (target: <100MB)")


if __name__ == "__main__":
    # Run tests with verbose output for Phase 3D validation
    unittest.main(verbosity=2)
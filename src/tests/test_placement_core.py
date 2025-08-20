#!/usr/bin/env python3
"""
Unit tests for Phase 3A: Core placement logic components
"""

import unittest
import numpy as np
from unittest.mock import Mock, patch
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from placement import (
    PlacementCommand, PlacementResult, PlacementStrategy,
    CandidateGenerator, BasicConstraintChecker, PlacementScorer, PlacementEngine,
    SimilarityEngine
)
from spatial import (
    Rectangle, ShelfSpaceManager, SimpleStabilityChecker, BasicReachabilityChecker
)


class TestDataStructures(unittest.TestCase):
    """Test basic data structures follow KISS principle"""
    
    def test_placement_command_creation(self):
        """Test PlacementCommand creation with essential fields only"""
        cmd = PlacementCommand(
            object_id="test_object",
            zone_id="shelf_0_zone_1", 
            position=(0.1, 0.2, 0.0),
            orientation=(0.0, 0.0, 0.0, 1.0),
            confidence_score=0.85
        )
        
        self.assertEqual(cmd.object_id, "test_object")
        self.assertEqual(cmd.zone_id, "shelf_0_zone_1")
        self.assertEqual(cmd.position, (0.1, 0.2, 0.0))
        self.assertEqual(cmd.confidence_score, 0.85)
    
    def test_placement_result_success(self):
        """Test successful placement result"""
        cmd = PlacementCommand("obj", "zone", (0,0,0), (0,0,0,1), 0.9)
        result = PlacementResult(success=True, placement_command=cmd)
        
        self.assertTrue(result.success)
        self.assertIsNotNone(result.placement_command)
        self.assertEqual(result.failure_reason, "")
    
    def test_placement_result_failure(self):
        """Test failure placement result"""
        result = PlacementResult(
            success=False, 
            placement_command=None,
            failure_reason="No zones available"
        )
        
        self.assertFalse(result.success)
        self.assertIsNone(result.placement_command)
        self.assertEqual(result.failure_reason, "No zones available")


class TestSpatialComponents(unittest.TestCase):
    """Test spatial analysis components"""
    
    def test_rectangle_operations(self):
        """Test Rectangle basic operations"""
        rect = Rectangle(1.0, 2.0, 3.0, 4.0)
        
        # Test point containment
        self.assertTrue(rect.contains_point(2.0, 3.0))
        self.assertFalse(rect.contains_point(0.5, 3.0))
        
        # Test overlap
        other = Rectangle(2.0, 3.0, 2.0, 2.0)
        self.assertTrue(rect.overlaps(other))
        
        no_overlap = Rectangle(10.0, 10.0, 1.0, 1.0)
        self.assertFalse(rect.overlaps(no_overlap))
    
    def test_shelf_space_manager(self):
        """Test ShelfSpaceManager free space detection"""
        manager = ShelfSpaceManager()
        
        # Set zone bounds
        zone_bounds = Rectangle(0.0, 0.0, 1.0, 1.0)
        manager.set_zone_bounds("test_zone", zone_bounds)
        
        # Should find free space in empty zone
        position = manager.find_free_space("test_zone", 0.2, 0.2)
        self.assertIsNotNone(position)
        
        # Add occupied space
        occupied = Rectangle(0.0, 0.0, 0.5, 0.5)
        manager.add_occupied_space("test_zone", occupied)
        
        # Should still find free space
        position = manager.find_free_space("test_zone", 0.2, 0.2)
        self.assertIsNotNone(position)
        
        # Fill up the zone - should not find space for large object
        for i in range(10):
            occupied = Rectangle(0.1 * i, 0.5, 0.1, 0.5)
            manager.add_occupied_space("test_zone", occupied)
        
        position = manager.find_free_space("test_zone", 0.8, 0.8)
        self.assertIsNone(position)
    
    def test_zone_efficiency_tracking(self):
        """Test Phase 3B: Zone efficiency tracking"""
        manager = ShelfSpaceManager()
        
        # Set up zone
        zone_bounds = Rectangle(0.0, 0.0, 1.0, 1.0)  # 1m x 1m zone
        manager.set_zone_bounds("test_zone", zone_bounds)
        
        # Initially empty
        self.assertEqual(manager.get_zone_efficiency("test_zone"), 0.0)
        
        # Add 25% occupied space
        occupied = Rectangle(0.0, 0.0, 0.5, 0.5)  # 0.25 area
        manager.add_occupied_space("test_zone", occupied)
        self.assertEqual(manager.get_zone_efficiency("test_zone"), 0.25)
        
        # Add more space to fill zone over 90%
        # Fill zone systematically to exceed 90%
        filled_rects = [
            Rectangle(0.5, 0.0, 0.3, 0.3),   # 9%
            Rectangle(0.8, 0.0, 0.2, 0.3),   # 6%
            Rectangle(0.5, 0.3, 0.5, 0.2),   # 10%
            Rectangle(0.0, 0.5, 1.0, 0.4),   # 40%
        ]
        
        for rect in filled_rects:
            manager.add_occupied_space("test_zone", rect)
        
        # Check efficiency triggers optimization (should be >=90%)
        efficiency = manager.get_zone_efficiency("test_zone")
        self.assertGreaterEqual(efficiency, 0.9)
        
        # Should skip search due to efficiency check
        position = manager.find_free_space("test_zone", 0.05, 0.05)
        self.assertIsNone(position)


class TestPlacementComponents(unittest.TestCase):
    """Test placement logic components follow SRP"""
    
    def setUp(self):
        """Set up test components"""
        self.space_manager = ShelfSpaceManager()
        self.space_manager.set_zone_bounds("zone_1", Rectangle(0, 0, 1, 1))
        self.space_manager.set_zone_bounds("zone_2", Rectangle(1, 0, 1, 1))
        
        self.candidate_generator = CandidateGenerator(self.space_manager)
        self.constraint_checker = BasicConstraintChecker(
            SimpleStabilityChecker(),
            BasicReachabilityChecker()
        )
        self.scorer = PlacementScorer()
    
    def test_candidate_generator(self):
        """Test CandidateGenerator follows SRP - only generates candidates"""
        similarity_scores = {"zone_1": 0.9, "zone_2": 0.7, "zone_3": 0.5}
        object_size = (0.1, 0.1)
        
        candidates = self.candidate_generator.generate(similarity_scores, object_size)
        
        # Should return candidates sorted by similarity
        self.assertGreater(len(candidates), 0)
        # Only zones with free space should be returned
        zone_ids = [zone_id for zone_id, _ in candidates]
        self.assertIn("zone_1", zone_ids)  # Has free space
        self.assertIn("zone_2", zone_ids)  # Has free space
        self.assertNotIn("zone_3", zone_ids)  # No bounds set, no free space
    
    def test_candidate_generator_efficiency_bonus(self):
        """Test Phase 3B: CandidateGenerator prefers less crowded zones"""
        similarity_scores = {"zone_1": 0.8, "zone_2": 0.8}  # Same similarity
        object_size = (0.1, 0.1)
        
        # Make zone_2 more crowded
        occupied = Rectangle(0.0, 0.0, 0.8, 0.8)  # 64% of zone area
        self.space_manager.add_occupied_space("zone_2", occupied)
        
        candidates = self.candidate_generator.generate(similarity_scores, object_size)
        
        # zone_1 should come first due to efficiency bonus (less crowded)
        zone_ids = [zone_id for zone_id, _ in candidates]
        self.assertEqual(zone_ids[0], "zone_1")
    
    def test_constraint_checker(self):
        """Test BasicConstraintChecker follows SRP - only checks constraints"""
        # Valid position
        self.assertTrue(
            self.constraint_checker.is_valid_position(
                "zone_1", (0.5, 0.5, 0.5), (0.1, 0.1)
            )
        )
        
        # Invalid position (too high for reachability)
        self.assertFalse(
            self.constraint_checker.is_valid_position(
                "zone_1", (0.5, 0.5, 3.0), (0.1, 0.1)
            )
        )
    
    def test_placement_scorer(self):
        """Test PlacementScorer follows SRP - only scores placements"""
        score = self.scorer.score_placement("zone_1", 0.85)
        self.assertEqual(score, 0.85)  # Simple pass-through for now
    
    def test_placement_scorer_strategies(self):
        """Test Phase 3C: PlacementScorer strategy bonuses"""
        from placement import PlacementStrategy
        
        base_score = 0.8
        
        # Test strategy bonuses
        stack_score = self.scorer.score_placement("zone_1", base_score, PlacementStrategy.STACK)
        self.assertAlmostEqual(stack_score, 0.85, places=6)  # +0.05 bonus
        
        lean_score = self.scorer.score_placement("zone_1", base_score, PlacementStrategy.LEAN)
        self.assertAlmostEqual(lean_score, 0.83, places=6)  # +0.03 bonus
        
        group_score = self.scorer.score_placement("zone_1", base_score, PlacementStrategy.GROUP)
        self.assertAlmostEqual(group_score, 0.82, places=6)  # +0.02 bonus
        
        place_score = self.scorer.score_placement("zone_1", base_score, PlacementStrategy.PLACE)
        self.assertAlmostEqual(place_score, 0.8, places=6)  # No bonus


class TestPlacementEngine(unittest.TestCase):
    """Test PlacementEngine integration"""
    
    def setUp(self):
        """Set up test environment"""
        self.similarity_engine = SimilarityEngine()
        self.similarity_engine.initialize()
        
        # Add some test zone embeddings
        for i in range(3):
            zone_id = f"shelf_0_zone_{i}"
            embedding = np.random.rand(768)  # DINOv3 embedding size
            embedding = embedding / np.linalg.norm(embedding)  # Normalize
            self.similarity_engine.store_zone_embedding(zone_id, embedding)
        
        self.placement_engine = PlacementEngine(self.similarity_engine)
        
        # Set up zone bounds in space manager
        for i in range(3):
            zone_id = f"shelf_0_zone_{i}"
            bounds = Rectangle(i * 1.0, 0.0, 1.0, 1.0)
            self.placement_engine.space_manager.set_zone_bounds(zone_id, bounds)
    
    def test_successful_placement(self):
        """Test successful placement pipeline"""
        # Create test object embedding
        object_embedding = np.random.rand(768)
        object_embedding = object_embedding / np.linalg.norm(object_embedding)
        
        result = self.placement_engine.find_placement(
            "test_object", object_embedding, (0.1, 0.1)
        )
        
        self.assertTrue(result.success)
        self.assertIsNotNone(result.placement_command)
        self.assertEqual(result.placement_command.object_id, "test_object")
        self.assertGreater(result.placement_command.confidence_score, 0)
    
    def test_no_zones_failure(self):
        """Test failure when no zones available"""
        # Empty similarity engine
        empty_engine = SimilarityEngine()
        empty_engine.initialize()
        placement_engine = PlacementEngine(empty_engine)
        
        object_embedding = np.random.rand(768)
        object_embedding = object_embedding / np.linalg.norm(object_embedding)
        
        result = placement_engine.find_placement("test_object", object_embedding)
        
        self.assertFalse(result.success)
        self.assertIsNone(result.placement_command)
        self.assertEqual(result.failure_reason, "No zones available")
    
    def test_no_free_space_failure(self):
        """Test failure when no free space available"""
        # Fill all zones
        for i in range(3):
            zone_id = f"shelf_0_zone_{i}"
            # Fill entire zone
            occupied = Rectangle(i * 1.0, 0.0, 1.0, 1.0)
            self.placement_engine.space_manager.add_occupied_space(zone_id, occupied)
        
        object_embedding = np.random.rand(768)
        object_embedding = object_embedding / np.linalg.norm(object_embedding)
        
        result = self.placement_engine.find_placement(
            "test_object", object_embedding, (0.1, 0.1)
        )
        
        self.assertFalse(result.success)
        self.assertEqual(result.failure_reason, "No valid placement found")
    
    def test_placement_with_strategies(self):
        """Test Phase 3C: Placement with different strategies"""
        from placement import PlacementStrategy
        
        # Create test object embedding
        object_embedding = np.random.rand(768)
        object_embedding = object_embedding / np.linalg.norm(object_embedding)
        
        # Test STACK strategy
        result = self.placement_engine.find_placement(
            "test_stack_object", object_embedding, (0.1, 0.1), PlacementStrategy.STACK
        )
        
        self.assertTrue(result.success)
        self.assertIsNotNone(result.placement_command)
        stack_score = result.placement_command.confidence_score
        
        # Test PLACE strategy for comparison
        result_place = self.placement_engine.find_placement(
            "test_place_object", object_embedding, (0.1, 0.1), PlacementStrategy.PLACE
        )
        place_score = result_place.placement_command.confidence_score
        
        # STACK should have higher score than PLACE due to strategy bonus
        self.assertGreater(stack_score, place_score)
        
        # Test LEAN strategy
        result = self.placement_engine.find_placement(
            "test_lean_object", object_embedding, (0.1, 0.1), PlacementStrategy.LEAN
        )
        
        self.assertTrue(result.success)
        self.assertIsNotNone(result.placement_command)


class TestIntegrationWithPhase2(unittest.TestCase):
    """Test integration with existing Phase 2 components"""
    
    def test_similarity_engine_compatibility(self):
        """Test PlacementEngine works with existing SimilarityEngine"""
        # This tests the interface compatibility
        similarity_engine = SimilarityEngine()
        placement_engine = PlacementEngine(similarity_engine)
        
        # Should initialize without errors
        placement_engine.initialize()
        
        # Should have proper state
        state = placement_engine.get_state()
        self.assertIn("similarity_engine_state", state)
        self.assertIn("space_manager_state", state)


if __name__ == "__main__":
    # Run tests with verbose output
    unittest.main(verbosity=2)
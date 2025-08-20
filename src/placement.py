#!/usr/bin/env python3
"""
Placement Logic
Visual similarity matching and constraint-based placement for shelf organization
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum
from interfaces import SimulationComponent
from spatial import ShelfSpaceManager, Rectangle, SimpleStabilityChecker, BasicReachabilityChecker


class PlacementStrategy(Enum):
    """Placement strategy types"""
    PLACE = "place"      # Standard placement on shelf surface
    STACK = "stack"      # Stack on top of existing objects  
    LEAN = "lean"        # Lean against shelf back wall
    GROUP = "group"      # Group with similar small items


@dataclass
class PlacementCommand:
    """Simple placement command - YAGNI: only essential fields"""
    object_id: str
    zone_id: str
    position: Tuple[float, float, float]
    orientation: Tuple[float, float, float, float]  # quaternion
    confidence_score: float


@dataclass 
class PlacementResult:
    """Simple placement result"""
    success: bool
    placement_command: Optional[PlacementCommand]
    failure_reason: str = ""


class SimilarityEngine(SimulationComponent):
    """Simple similarity-based placement decisions"""
    
    def __init__(self):
        self.object_embeddings = {}  # object_id -> embedding
        self.zone_embeddings = {}    # zone_id -> embedding
        
    def initialize(self, use_gui: bool = False) -> None:
        """Initialize similarity engine"""
        pass
        
    def store_object_embedding(self, object_id: str, embedding: np.ndarray) -> None:
        """Store object feature embedding"""
        self.object_embeddings[object_id] = embedding
        
    def store_zone_embedding(self, zone_id: str, embedding: np.ndarray) -> None:
        """Store zone feature embedding"""
        self.zone_embeddings[zone_id] = embedding
        
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute cosine similarity between embeddings"""
        return np.dot(embedding1, embedding2)  # Already normalized
        
    def find_best_zone(self, object_embedding: np.ndarray) -> Optional[str]:
        """Find best zone for object based on visual similarity"""
        if not self.zone_embeddings:
            return None
            
        best_zone = None
        best_score = -1.0
        
        for zone_id, zone_embedding in self.zone_embeddings.items():
            score = self.compute_similarity(object_embedding, zone_embedding)
            if score > best_score:
                best_score = score
                best_zone = zone_id
                
        return best_zone
        
    def get_similarity_heatmap(self, object_embedding: np.ndarray) -> Dict[str, float]:
        """Get similarity scores for all zones"""
        heatmap = {}
        for zone_id, zone_embedding in self.zone_embeddings.items():
            heatmap[zone_id] = self.compute_similarity(object_embedding, zone_embedding)
        return heatmap
        
    def cleanup(self) -> None:
        """Clean up resources"""
        self.object_embeddings.clear()
        self.zone_embeddings.clear()
        
    def get_state(self) -> dict:
        """Get current state"""
        return {
            "num_objects": len(self.object_embeddings),
            "num_zones": len(self.zone_embeddings)
        }


class CandidateGenerator:
    """SRP: Single responsibility - generate placement candidates"""
    
    def __init__(self, space_manager: ShelfSpaceManager):
        self.space_manager = space_manager
    
    def generate(self, similarity_scores: Dict[str, float], 
                object_size: Tuple[float, float]) -> List[Tuple[str, Tuple[float, float]]]:
        """Generate candidate positions with size-adaptive sampling"""
        candidates = []
        
        # Sort zones by similarity with efficiency bonus (Phase 3B optimization)
        def zone_score(zone_id):
            similarity = similarity_scores[zone_id]
            efficiency = self.space_manager.get_zone_efficiency(zone_id)
            # Small bonus for less crowded zones (up to 10% boost)
            efficiency_bonus = (1.0 - efficiency) * 0.1
            return similarity + efficiency_bonus
        
        sorted_zones = sorted(similarity_scores.keys(),
                            key=zone_score,
                            reverse=True)
        
        # KISS: Find first valid position in each zone
        for zone_id in sorted_zones:
            position = self.space_manager.find_free_space(
                zone_id, object_size[0], object_size[1]
            )
            if position:
                candidates.append((zone_id, position))
                
        return candidates


class BasicConstraintChecker:
    """SRP: Single responsibility - check basic constraints"""
    
    def __init__(self, stability_checker: SimpleStabilityChecker, 
                 reachability_checker: BasicReachabilityChecker):
        self.stability_checker = stability_checker
        self.reachability_checker = reachability_checker
    
    def is_valid_position(self, zone_id: str, position: Tuple[float, float, float],
                         object_size: Tuple[float, float]) -> bool:
        """Check if position meets basic constraints"""
        # Check reachability
        if not self.reachability_checker.is_reachable(zone_id, position):
            return False
            
        # Check stability
        if not self.stability_checker.check_support(position[:2], object_size):
            return False
            
        return True


class PlacementScorer:
    """SRP: Single responsibility - score placement options"""
    
    def score_placement(self, zone_id: str, similarity_score: float, 
                       strategy: PlacementStrategy = PlacementStrategy.PLACE) -> float:
        """Score a placement option with strategy consideration"""
        # KISS: Base score is similarity, small strategy bonus
        base_score = similarity_score
        
        # Simple strategy bonuses (Phase 3C)
        if strategy == PlacementStrategy.STACK:
            return base_score + 0.05  # Small bonus for stacking
        elif strategy == PlacementStrategy.LEAN:
            return base_score + 0.03  # Small bonus for leaning
        elif strategy == PlacementStrategy.GROUP:
            return base_score + 0.02  # Small bonus for grouping
        
        return base_score


class PlacementEngine(SimulationComponent):
    """SRP: Single responsibility - orchestrate placement decision"""
    
    def __init__(self, similarity_engine: SimilarityEngine):
        # DI: Dependency injection for testability
        self.similarity_engine = similarity_engine
        self.space_manager = ShelfSpaceManager()
        self.candidate_generator = CandidateGenerator(self.space_manager)
        self.constraint_checker = BasicConstraintChecker(
            SimpleStabilityChecker(), 
            BasicReachabilityChecker()
        )
        self.scorer = PlacementScorer()
    
    def initialize(self, use_gui: bool = False) -> None:
        """Initialize placement engine"""
        pass
    
    def find_placement(self, object_id: str, object_embedding: np.ndarray, 
                      object_size: Tuple[float, float] = (0.1, 0.1),
                      preferred_strategy: PlacementStrategy = PlacementStrategy.PLACE) -> PlacementResult:
        """
        Find best placement for object with strategy consideration
        KISS: Simple pipeline without over-engineering
        """
        # Get similarity scores from Phase 2
        similarity_scores = self.similarity_engine.get_similarity_heatmap(object_embedding)
        
        if not similarity_scores:
            return PlacementResult(
                success=False,
                placement_command=None,
                failure_reason="No zones available"
            )
        
        # Generate candidates with size-adaptive sampling
        candidates = self.candidate_generator.generate(similarity_scores, object_size)
        
        # Try preferred strategy first, then fallback to PLACE
        strategies_to_try = [preferred_strategy]
        if preferred_strategy != PlacementStrategy.PLACE:
            strategies_to_try.append(PlacementStrategy.PLACE)
        
        for strategy in strategies_to_try:
            for zone_id, position_2d in candidates:
                position_3d = self._apply_strategy(strategy, position_2d, object_size, zone_id)
                
                if self.constraint_checker.is_valid_position(zone_id, position_3d, object_size):
                    score = self.scorer.score_placement(zone_id, similarity_scores[zone_id], strategy)
                    
                    placement_command = PlacementCommand(
                        object_id=object_id,
                        zone_id=zone_id,
                        position=position_3d,
                        orientation=(0.0, 0.0, 0.0, 1.0),  # Identity quaternion
                        confidence_score=score
                    )
                    
                    return PlacementResult(
                        success=True,
                        placement_command=placement_command
                    )
        
        return PlacementResult(
            success=False,
            placement_command=None,
            failure_reason="No valid placement found"
        )
    
    def _apply_strategy(self, strategy: PlacementStrategy, position_2d: Tuple[float, float], 
                       object_size: Tuple[float, float], zone_id: str) -> Tuple[float, float, float]:
        """Apply placement strategy to determine final 3D position"""
        x, y = position_2d
        
        if strategy == PlacementStrategy.STACK:
            # KISS: Simple stacking - check if zone has existing objects, stack on top
            efficiency = self.space_manager.get_zone_efficiency(zone_id)
            if efficiency > 0.3:  # If zone has some objects, try to stack
                z = 0.15  # Simple 15cm stacking height
            else:
                z = 0.0  # Ground level if zone is mostly empty
                
        elif strategy == PlacementStrategy.LEAN:
            # KISS: Simple leaning - push object toward back of zone
            zone_bounds = self.space_manager.zone_bounds.get(zone_id)
            if zone_bounds:
                y = zone_bounds.y + zone_bounds.height - object_size[1] - 0.02  # 2cm from back wall
            z = 0.0  # Ground level
            
        elif strategy == PlacementStrategy.GROUP:
            # KISS: Simple grouping - try to cluster objects together  
            efficiency = self.space_manager.get_zone_efficiency(zone_id)
            if efficiency > 0.2:  # If objects exist, try to get closer to them
                # Simple heuristic - move toward center of occupied space
                zone_bounds = self.space_manager.zone_bounds.get(zone_id)
                if zone_bounds:
                    center_x = zone_bounds.x + zone_bounds.width / 2
                    center_y = zone_bounds.y + zone_bounds.height / 2
                    # Move 20% toward center
                    x = 0.8 * x + 0.2 * center_x
                    y = 0.8 * y + 0.2 * center_y
            z = 0.0
            
        else:  # PlacementStrategy.PLACE (default)
            z = 0.0  # Ground level placement
        
        return (x, y, z)
    
    def _get_zone_center(self, zone_id: str) -> Tuple[float, float, float]:
        """Get center position of zone - placeholder"""
        # YAGNI: Simple implementation for now
        return (0.0, 0.0, 0.0)
    
    def cleanup(self) -> None:
        """Clean up resources"""
        pass
    
    def get_state(self) -> dict:
        """Get current state"""
        return {
            "similarity_engine_state": self.similarity_engine.get_state(),
            "space_manager_state": self.space_manager.get_state()
        }
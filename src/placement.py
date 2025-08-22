#!/usr/bin/env python3
"""
Placement Logic
Visual similarity matching and constraint-based placement for shelf organization
"""

import numpy as np
import pybullet as p
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
        # Ensure embeddings are 1D to prevent array return from np.dot
        emb1_flat = embedding1.flatten()
        emb2_flat = embedding2.flatten()
        result = np.dot(emb1_flat, emb2_flat)
        return float(result)  # Ensure scalar return
        
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
        self.physics_client = None  # Will be set when physics client is available
    
    def initialize(self, use_gui: bool = False) -> None:
        """Initialize placement engine"""
        pass
    
    def set_physics_client(self, physics_client) -> None:
        """Set physics client for collision detection"""
        self.physics_client = physics_client
    
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
                position_3d = self._apply_strategy(strategy, position_2d, object_size, zone_id, object_id)
                
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
                       object_size: Tuple[float, float], zone_id: str, object_id: str = "") -> Tuple[float, float, float]:
        """Apply placement strategy to determine final 3D position"""
        x, y = position_2d
        
        if strategy == PlacementStrategy.STACK:
            # Real stacking with collision detection and stability analysis
            z = self._find_stacking_height(x, y, object_size, object_id)
                
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
    
    def _find_stacking_height(self, x: float, y: float, object_size: Tuple[float, float], 
                            object_id: str = "") -> float:
        """Find proper stacking height using real collision detection and stability analysis"""
        if self.physics_client is None:
            # Fallback to ground level if no physics client
            return 0.0
            
        # For stacking, we want to find objects DIRECTLY below this position
        # Use a small search radius to find objects at this exact location
        search_radius = max(object_size) * 0.8  # 80% overlap - we want to stack directly on top
        
        # Get all objects in the vicinity using PyBullet collision detection
        objects_below = self._get_objects_in_area(x, y, search_radius)
        
        if not objects_below:
            return 0.0  # Ground level placement
            
        # Check stacking stability - heavy objects should not stack on delicate ones
        if not self._is_stable_stacking(object_id, objects_below):
            return 0.0  # Reject stacking, place on ground instead
            
        # Find the highest point among all objects below
        highest_point = self._calculate_highest_surface(objects_below, x, y, object_size)
        
        if highest_point <= 0.0:
            return 0.0  # No suitable surface found
            
        # Add object height/2 + small clearance for stable stacking
        object_height = object_size[0]  # Assume cubic objects for simplicity
        clearance = 0.01  # 1cm clearance
        
        return highest_point + object_height/2 + clearance
    
    def _get_objects_in_area(self, x: float, y: float, radius: float) -> List[int]:
        """Get all PyBullet objects within radius of target position"""
        if self.physics_client is None:
            return []
            
        objects_in_area = []
        
        try:
            # Get all bodies in the simulation
            num_bodies = p.getNumBodies(physicsClientId=self.physics_client)
            
            for body_id in range(num_bodies):
                # Skip ground plane (body_id = 0) and robot
                if body_id <= 2:  # Ground plane and robot are typically first bodies
                    continue
                    
                # Get object position
                pos, _ = p.getBasePositionAndOrientation(body_id, physicsClientId=self.physics_client)
                obj_x, obj_y, obj_z = pos
                
                # Check if object is in horizontal range and above ground
                distance = np.sqrt((obj_x - x)**2 + (obj_y - y)**2)
                if distance <= radius and obj_z >= 0.0:
                    objects_in_area.append(body_id)
                    
        except Exception:
            # If PyBullet query fails, return empty list
            pass
            
        return objects_in_area
    
    def _calculate_highest_surface(self, object_ids: List[int], target_x: float, 
                                 target_y: float, object_size: Tuple[float, float]) -> float:
        """Calculate the highest surface point for stacking"""
        if not object_ids or self.physics_client is None:
            return 0.0
            
        highest_z = 0.0
        
        try:
            for body_id in object_ids:
                # Get object bounding box
                aabb_min, aabb_max = p.getAABB(body_id, physicsClientId=self.physics_client)
                
                # Check if target position would overlap with this object horizontally
                obj_min_x, obj_min_y, _ = aabb_min
                obj_max_x, obj_max_y, obj_max_z = aabb_max
                
                # Check for horizontal overlap considering object size
                target_min_x = target_x - object_size[0]/2
                target_max_x = target_x + object_size[0]/2
                target_min_y = target_y - object_size[1]/2
                target_max_y = target_y + object_size[1]/2
                
                # Check if bounding boxes overlap horizontally
                x_overlap = not (target_max_x < obj_min_x or target_min_x > obj_max_x)
                y_overlap = not (target_max_y < obj_min_y or target_min_y > obj_max_y)
                
                if x_overlap and y_overlap:
                    # This object is directly below our target position
                    highest_z = max(highest_z, obj_max_z)
                    
        except Exception:
            # If bounding box query fails, return current highest
            pass
            
        return highest_z
    
    def _is_stable_stacking(self, object_id: str, objects_below: List[int]) -> bool:
        """Check if stacking this object on objects below is stable"""
        if not object_id or not objects_below:
            return True  # Default to allowing stacking
            
        # Get object weight/stability characteristics
        object_weight = self._get_object_weight_category(object_id)
        
        # Check weight categories of objects below
        for body_id in objects_below:
            try:
                # Get the body's user data or use heuristics to determine stability
                base_object_weight = self._get_body_weight_category(body_id)
                
                # Heavy objects should not stack on light/delicate ones
                if object_weight == "heavy" and base_object_weight in ["light", "delicate"]:
                    return False
                    
                # Very heavy objects (like 12-packs) should only stack on sturdy objects
                if object_weight == "very_heavy" and base_object_weight != "sturdy":
                    return False
                    
            except Exception:
                # If we can't determine weight, be conservative
                if "pack" in object_id.lower() or "12" in object_id:
                    return False  # Don't stack heavy packs on unknown objects
                    
        return True
    
    def _get_object_weight_category(self, object_id: str) -> str:
        """Categorize object weight based on object ID"""
        object_lower = object_id.lower()
        
        # Very heavy objects
        if any(term in object_lower for term in ["12_pack", "pack_12", "cola12", "soda_cola"]):
            return "very_heavy"
            
        # Heavy objects
        if any(term in object_lower for term in ["pressure_cooker", "frypan", "pot", "pan"]):
            return "heavy"
            
        # Delicate objects
        if any(term in object_lower for term in ["mug", "cup", "bowl", "porcelain", "teapot"]):
            return "delicate"
            
        # Light objects
        if any(term in object_lower for term in ["spoon", "spatula", "utensil"]):
            return "light"
            
        # Default to medium weight
        return "medium"
    
    def _get_body_weight_category(self, body_id: int) -> str:
        """Get weight category of PyBullet body (simplified heuristic)"""
        if self.physics_client is None:
            return "medium"
            
        try:
            # Get object mass (if available)
            dynamics_info = p.getDynamicsInfo(body_id, -1, physicsClientId=self.physics_client)
            mass = dynamics_info[0]  # Mass is first element
            
            # Categorize based on mass
            if mass > 2.0:  # > 2kg
                return "very_heavy"
            elif mass > 1.0:  # > 1kg
                return "heavy" 
            elif mass > 0.5:  # > 0.5kg
                return "medium"
            elif mass > 0.1:  # > 0.1kg
                return "light"
            else:
                return "delicate"
                
        except Exception:
            # If mass query fails, use size as proxy
            try:
                aabb_min, aabb_max = p.getAABB(body_id, physicsClientId=self.physics_client)
                size = np.linalg.norm(np.array(aabb_max) - np.array(aabb_min))
                
                if size > 0.4:  # Large objects are likely heavy
                    return "sturdy"
                elif size > 0.2:
                    return "medium"
                else:
                    return "delicate"
            except Exception:
                return "medium"
    
    def _get_zone_center(self, zone_id: str) -> Tuple[float, float, float]:
        """Get center position of zone - placeholder"""
        # Simple implementation for now
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
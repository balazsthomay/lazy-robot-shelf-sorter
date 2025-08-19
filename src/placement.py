#!/usr/bin/env python3
"""
Placement Logic
Simple visual similarity matching for shelf organization
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from interfaces import SimulationComponent


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
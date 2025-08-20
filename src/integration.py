#!/usr/bin/env python3
"""
Phase Boundary Management
Manages integration between Phase 2 (Vision) → Phase 3 (Placement) → Phase 4 (Control)
"""

import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from interfaces import SimulationComponent
from placement import PlacementEngine, PlacementCommand, PlacementResult, PlacementStrategy


class FallbackReason(Enum):
    """Reasons for placement fallback"""
    TIMEOUT = "timeout"
    NO_VALID_PLACEMENT = "no_valid_placement"
    PHYSICS_FAILURE = "physics_failure"
    REACHABILITY_FAILURE = "reachability_failure"
    STABILITY_FAILURE = "stability_failure"


@dataclass
class PlacementMetrics:
    """Performance and quality metrics for placement decisions"""
    # Timing
    total_time_ms: float = 0.0
    candidate_generation_ms: float = 0.0
    constraint_checking_ms: float = 0.0
    strategy_application_ms: float = 0.0
    
    # Throughput
    candidates_generated: int = 0
    candidates_filtered: int = 0
    strategies_attempted: int = 0
    
    # Quality
    chosen_similarity_score: float = 0.0
    chosen_confidence_score: float = 0.0
    fallback_steps_taken: int = 0
    
    # Failure Analysis
    failure_reasons: List[str] = field(default_factory=list)
    constraints_violated: List[str] = field(default_factory=list)


@dataclass
class FallbackEntry:
    """Entry in fallback blacklist"""
    zone_id: str
    strategy: PlacementStrategy
    failure_reason: FallbackReason
    timestamp: float
    penalty_decay_time: float = 300.0  # 5 minutes


class FallbackManager:
    """SRP: Manage placement fallbacks with time budgets and blacklists"""
    
    def __init__(self, time_budget_seconds: float = 30.0):
        self.time_budget = time_budget_seconds
        self.blacklist: List[FallbackEntry] = []
        self.fallback_chain = [
            PlacementStrategy.PLACE,
            PlacementStrategy.LEAN,
            PlacementStrategy.GROUP,
            PlacementStrategy.STACK  # Most complex last
        ]
    
    def is_blacklisted(self, zone_id: str, strategy: PlacementStrategy) -> bool:
        """Check if zone-strategy combination is blacklisted"""
        current_time = time.time()
        
        # Clean expired blacklist entries
        self.blacklist = [
            entry for entry in self.blacklist 
            if (current_time - entry.timestamp) < entry.penalty_decay_time
        ]
        
        # Check for active blacklist entries
        for entry in self.blacklist:
            if entry.zone_id == zone_id and entry.strategy == strategy:
                return True
        return False
    
    def add_to_blacklist(self, zone_id: str, strategy: PlacementStrategy, 
                        reason: FallbackReason) -> None:
        """Add failed zone-strategy to blacklist"""
        entry = FallbackEntry(
            zone_id=zone_id,
            strategy=strategy,
            failure_reason=reason,
            timestamp=time.time()
        )
        self.blacklist.append(entry)
    
    def get_fallback_strategies(self, preferred_strategy: PlacementStrategy) -> List[PlacementStrategy]:
        """Get ordered list of strategies to try, starting with preferred"""
        strategies = [preferred_strategy]
        
        # Add remaining strategies in fallback order
        for strategy in self.fallback_chain:
            if strategy != preferred_strategy and strategy not in strategies:
                strategies.append(strategy)
        
        return strategies
    
    def should_continue(self, start_time: float) -> bool:
        """Check if we should continue trying (within time budget)"""
        elapsed = time.time() - start_time
        return elapsed < self.time_budget
    
    def get_remaining_time(self, start_time: float) -> float:
        """Get remaining time budget"""
        elapsed = time.time() - start_time
        return max(0.0, self.time_budget - elapsed)


class IntegratedPlacementEngine(SimulationComponent):
    """SRP: Production-ready placement engine with fallback management"""
    
    def __init__(self, similarity_engine, time_budget_seconds: float = 30.0):
        self.placement_engine = PlacementEngine(similarity_engine)
        self.fallback_manager = FallbackManager(time_budget_seconds)
        self.metrics_history: List[PlacementMetrics] = []
    
    def initialize(self, use_gui: bool = False) -> None:
        """Initialize integrated placement engine"""
        self.placement_engine.initialize(use_gui)
    
    def find_placement_with_fallback(self, object_id: str, object_embedding: np.ndarray,
                                   object_size: tuple = (0.1, 0.1),
                                   preferred_strategy: PlacementStrategy = PlacementStrategy.PLACE) -> PlacementResult:
        """
        Find placement with comprehensive fallback management
        SOLID: Single responsibility for production placement decisions
        """
        start_time = time.time()
        metrics = PlacementMetrics()
        
        try:
            # Get similarity scores from Phase 2 integration
            similarity_scores = self.placement_engine.similarity_engine.get_similarity_heatmap(object_embedding)
            
            if not similarity_scores:
                metrics.failure_reasons.append("no_zones_available")
                return self._create_failure_result("No zones available from Phase 2", metrics)
            
            # Try strategies in fallback order
            strategies_to_try = self.fallback_manager.get_fallback_strategies(preferred_strategy)
            
            for strategy in strategies_to_try:
                if not self.fallback_manager.should_continue(start_time):
                    metrics.failure_reasons.append("timeout")
                    break
                
                metrics.strategies_attempted += 1
                strategy_start = time.time()
                
                # Try placement with current strategy
                result = self._try_strategy(object_id, object_embedding, object_size, 
                                          strategy, similarity_scores, metrics)
                
                strategy_time = (time.time() - strategy_start) * 1000
                metrics.strategy_application_ms += strategy_time
                
                if result.success:
                    # Success! Record metrics and return
                    metrics.total_time_ms = (time.time() - start_time) * 1000
                    metrics.chosen_similarity_score = result.placement_command.confidence_score
                    metrics.chosen_confidence_score = result.placement_command.confidence_score
                    self.metrics_history.append(metrics)
                    return result
                else:
                    # Strategy failed, add to blacklist and try next
                    # Extract zone from failure if available  
                    if hasattr(result, 'attempted_zone'):
                        self.fallback_manager.add_to_blacklist(
                            result.attempted_zone, strategy, FallbackReason.NO_VALID_PLACEMENT
                        )
                    metrics.fallback_steps_taken += 1
            
            # All strategies exhausted
            metrics.total_time_ms = (time.time() - start_time) * 1000
            metrics.failure_reasons.append("all_strategies_exhausted")
            self.metrics_history.append(metrics)
            
            return PlacementResult(
                success=False,
                placement_command=None,
                failure_reason=f"All {len(strategies_to_try)} strategies failed within {self.fallback_manager.time_budget}s"
            )
            
        except Exception as e:
            metrics.total_time_ms = (time.time() - start_time) * 1000
            metrics.failure_reasons.append(f"exception_{type(e).__name__}")
            self.metrics_history.append(metrics)
            
            return PlacementResult(
                success=False,
                placement_command=None,
                failure_reason=f"Integration error: {str(e)}"
            )
    
    def _try_strategy(self, object_id: str, object_embedding: np.ndarray, object_size: tuple,
                     strategy: PlacementStrategy, similarity_scores: Dict[str, float],
                     metrics: PlacementMetrics) -> PlacementResult:
        """Try a specific strategy with blacklist filtering"""
        
        # Filter out blacklisted zone-strategy combinations
        available_zones = {}
        for zone_id, score in similarity_scores.items():
            if not self.fallback_manager.is_blacklisted(zone_id, strategy):
                available_zones[zone_id] = score
        
        if not available_zones:
            return PlacementResult(
                success=False,
                placement_command=None,
                failure_reason=f"All zones blacklisted for strategy {strategy.value}"
            )
        
        # Use existing placement engine with filtered zones
        # Simulate zone restriction by temporarily modifying similarity engine
        original_zones = self.placement_engine.similarity_engine.zone_embeddings.copy()
        
        # Filter zones
        filtered_zones = {
            zone_id: embedding for zone_id, embedding 
            in original_zones.items() if zone_id in available_zones
        }
        self.placement_engine.similarity_engine.zone_embeddings = filtered_zones
        
        try:
            result = self.placement_engine.find_placement(
                object_id, object_embedding, object_size, strategy
            )
            return result
        finally:
            # Restore original zones
            self.placement_engine.similarity_engine.zone_embeddings = original_zones
    
    def _create_failure_result(self, reason: str, metrics: PlacementMetrics) -> PlacementResult:
        """Create standardized failure result"""
        self.metrics_history.append(metrics)
        return PlacementResult(
            success=False,
            placement_command=None, 
            failure_reason=reason
        )
    
    def get_performance_metrics(self, last_n: int = 10) -> Dict[str, Any]:
        """Get performance statistics for recent placements"""
        recent_metrics = self.metrics_history[-last_n:]
        
        if not recent_metrics:
            return {"error": "No metrics available"}
        
        return {
            "placement_count": len(recent_metrics),
            "average_time_ms": np.mean([m.total_time_ms for m in recent_metrics]),
            "success_rate": len([m for m in recent_metrics if not m.failure_reasons]) / len(recent_metrics),
            "average_fallback_steps": np.mean([m.fallback_steps_taken for m in recent_metrics]),
            "common_failures": self._get_failure_summary(recent_metrics),
            "performance_percentiles": {
                "p50_ms": np.percentile([m.total_time_ms for m in recent_metrics], 50),
                "p90_ms": np.percentile([m.total_time_ms for m in recent_metrics], 90),
                "p99_ms": np.percentile([m.total_time_ms for m in recent_metrics], 99),
            }
        }
    
    def _get_failure_summary(self, metrics_list: List[PlacementMetrics]) -> Dict[str, int]:
        """Summarize failure reasons across metrics"""
        failure_counts = {}
        for metrics in metrics_list:
            for reason in metrics.failure_reasons:
                failure_counts[reason] = failure_counts.get(reason, 0) + 1
        return failure_counts
    
    def cleanup(self) -> None:
        """Clean up resources"""
        self.placement_engine.cleanup()
        self.blacklist.clear()
        self.metrics_history.clear()
    
    def get_state(self) -> dict:
        """Get current state for debugging"""
        return {
            "placement_engine_state": self.placement_engine.get_state(),
            "blacklist_entries": len(self.fallback_manager.blacklist),
            "metrics_history_size": len(self.metrics_history),
            "time_budget_seconds": self.fallback_manager.time_budget
        }


# Phase 2-3-4 Integration Interface
class PhaseBoundaryManager:
    """Manages data flow and error handling between phases"""
    
    def __init__(self, similarity_engine, placement_engine: IntegratedPlacementEngine):
        self.similarity_engine = similarity_engine  # Phase 2
        self.placement_engine = placement_engine      # Phase 3
        self.world_state_version = 0
    
    def process_object_placement(self, object_id: str, object_data: Dict[str, Any]) -> PlacementResult:
        """
        Complete Phase 2→3→4 pipeline
        SOLID: Single responsibility for cross-phase coordination
        """
        try:
            # Phase 2: Extract visual features
            if 'embedding' in object_data:
                object_embedding = object_data['embedding']
            else:
                # Fallback: extract features from image data
                object_embedding = self._extract_features(object_data)
            
            # Phase 3: Find placement
            object_size = object_data.get('size', (0.1, 0.1))
            preferred_strategy = object_data.get('strategy', PlacementStrategy.PLACE)
            
            placement_result = self.placement_engine.find_placement_with_fallback(
                object_id, object_embedding, object_size, preferred_strategy
            )
            
            if placement_result.success:
                # Update world state for next iteration
                self.world_state_version += 1
                
                # Phase 4 Interface: Add motion planning metadata
                placement_result.placement_command.world_state_version = self.world_state_version
            
            return placement_result
            
        except Exception as e:
            return PlacementResult(
                success=False,
                placement_command=None,
                failure_reason=f"Phase boundary error: {str(e)}"
            )
    
    def _extract_features(self, object_data: Dict[str, Any]) -> np.ndarray:
        """Fallback feature extraction if embedding not provided"""
        if 'image' not in object_data:
            raise ValueError("No embedding or image data provided for feature extraction")
        # Placeholder for Phase 2 integration
        return np.random.rand(768)  # DINOv3 embedding size
    
    def get_cross_phase_metrics(self) -> Dict[str, Any]:
        """Get metrics across all phases"""
        return {
            "world_state_version": self.world_state_version,
            "phase_2_state": self.similarity_engine.get_state(),
            "phase_3_metrics": self.placement_engine.get_performance_metrics(),
        }
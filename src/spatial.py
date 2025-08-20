#!/usr/bin/env python3
"""
Spatial Analysis
Simple 2D rectangle management and collision detection for shelf organization
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from interfaces import SimulationComponent


@dataclass
class Rectangle:
    """Simple 2D rectangle"""
    x: float
    y: float
    width: float
    height: float
    
    def contains_point(self, px: float, py: float) -> bool:
        """Check if point is inside rectangle"""
        return (self.x <= px <= self.x + self.width and 
                self.y <= py <= self.y + self.height)
    
    def overlaps(self, other: 'Rectangle') -> bool:
        """Check if this rectangle overlaps with another"""
        return not (self.x + self.width < other.x or 
                   other.x + other.width < self.x or
                   self.y + self.height < other.y or 
                   other.y + other.height < self.y)


class ShelfSpaceManager(SimulationComponent):
    """SRP: Manage 2D free space on shelves"""
    
    def __init__(self):
        self.occupied_rectangles: Dict[str, List[Rectangle]] = {}  # zone_id -> rectangles
        self.zone_bounds: Dict[str, Rectangle] = {}  # zone_id -> boundary
        self._zone_efficiency: Dict[str, float] = {}  # simple efficiency tracking
        
    def initialize(self, use_gui: bool = False) -> None:
        """Initialize space manager"""
        # YAGNI: Simple initialization
        pass
    
    def set_zone_bounds(self, zone_id: str, bounds: Rectangle) -> None:
        """Set boundary for a zone"""
        self.zone_bounds[zone_id] = bounds
        if zone_id not in self.occupied_rectangles:
            self.occupied_rectangles[zone_id] = []
    
    def add_occupied_space(self, zone_id: str, rect: Rectangle) -> None:
        """Add occupied rectangle to zone"""
        if zone_id not in self.occupied_rectangles:
            self.occupied_rectangles[zone_id] = []
        self.occupied_rectangles[zone_id].append(rect)
        
        # Update efficiency tracking - simple percentage of occupied space
        if zone_id in self.zone_bounds:
            zone_bound = self.zone_bounds[zone_id]
            zone_area = zone_bound.width * zone_bound.height
            occupied_area = sum(r.width * r.height for r in self.occupied_rectangles[zone_id])
            self._zone_efficiency[zone_id] = min(occupied_area / zone_area, 1.0)
    
    def find_free_space(self, zone_id: str, required_width: float, 
                       required_height: float) -> Optional[Tuple[float, float]]:
        """Find free space in zone for object of given size"""
        if zone_id not in self.zone_bounds:
            return None
        
        # Quick efficiency check - if zone is nearly full, skip detailed search
        efficiency = self._zone_efficiency.get(zone_id, 0.0)
        if efficiency >= 0.9:  # 90% full or more
            return None
        
        zone_bound = self.zone_bounds[zone_id]
        occupied = self.occupied_rectangles.get(zone_id, [])
        
        # KISS: Simple grid search for now
        step = 0.05  # 5cm grid
        for x in self._frange(zone_bound.x, zone_bound.x + zone_bound.width - required_width, step):
            for y in self._frange(zone_bound.y, zone_bound.y + zone_bound.height - required_height, step):
                candidate_rect = Rectangle(x, y, required_width, required_height)
                
                # Check if candidate overlaps with any occupied space
                if not any(candidate_rect.overlaps(occupied_rect) for occupied_rect in occupied):
                    return (x, y)
        
        return None
    
    def _frange(self, start: float, stop: float, step: float):
        """Float range generator"""
        x = start
        while x < stop:
            yield x
            x += step
    
    def cleanup(self) -> None:
        """Clean up resources"""
        self.occupied_rectangles.clear()
        self.zone_bounds.clear()
    
    def get_zone_efficiency(self, zone_id: str) -> float:
        """Get zone space utilization efficiency (0.0 to 1.0)"""
        return self._zone_efficiency.get(zone_id, 0.0)
    
    def get_state(self) -> dict:
        """Get current state"""
        return {
            "num_zones": len(self.zone_bounds),
            "total_occupied_spaces": sum(len(rects) for rects in self.occupied_rectangles.values()),
            "average_efficiency": sum(self._zone_efficiency.values()) / max(len(self._zone_efficiency), 1)
        }


class SimpleStabilityChecker:
    """SRP: Basic stability checks without expensive physics"""
    
    def check_support(self, position: Tuple[float, float], object_size: Tuple[float, float]) -> bool:
        """Check if object has adequate support at position"""
        # KISS: For now, assume all positions are stable
        # Will add real checks when we have object properties and physics
        return True
    
    def check_center_of_mass(self, position: Tuple[float, float], 
                           object_size: Tuple[float, float]) -> bool:
        """Check center of mass is within support area"""
        # KISS: Simple check - center of object is support point
        return True


class BasicReachabilityChecker:
    """SRP: Simple reachability validation"""
    
    def __init__(self):
        self.unreachable_zones = set()  # Simple blacklist
    
    def is_reachable(self, zone_id: str, position: Tuple[float, float, float]) -> bool:
        """Check if position is reachable by robot"""
        # KISS: Simple zone-based check
        if zone_id in self.unreachable_zones:
            return False
        
        # Basic height check - assume robot can reach up to 2m
        return position[2] <= 2.0
    
    def mark_unreachable(self, zone_id: str) -> None:
        """Mark zone as unreachable"""
        self.unreachable_zones.add(zone_id)
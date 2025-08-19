#!/usr/bin/env python3
"""
API Contracts - Simple interfaces for components
Part of Phase 1: Foundation - Milestone 2

"""

from abc import ABC, abstractmethod
from typing import List, Tuple


class SimulationComponent(ABC):
    """Base interface for all simulation components"""
    
    @abstractmethod
    def initialize(self) -> None:
        """Initialize component"""
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """Clean up resources"""
        pass


class EnvironmentProvider(ABC):
    """Only shelf position interface for now"""
    
    @abstractmethod
    def get_shelf_positions(self) -> List[Tuple[float, float, float]]:
        """Get all shelf positions"""
        pass


# DRY: Simple validation function
def validate_interfaces():
    """Verify interfaces work"""
    from simulation import ShelfEnvironment, ShelfConfiguration
    
    # Test interface compliance
    config = ShelfConfiguration(num_shelves=1)
    env = ShelfEnvironment(config)
    
    # Check if ShelfEnvironment follows contracts
    assert hasattr(env, 'initialize')
    assert hasattr(env, 'cleanup') 
    assert hasattr(env, 'get_shelf_positions')
    
    print("✅ API contracts validated")


if __name__ == "__main__":
    validate_interfaces()
#!/usr/bin/env python3
"""
Debug clearance checking
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from control import ShelfGeometry

def debug_clearance():
    """Debug the clearance checking logic"""
    shelf_positions = [(0.0, 0.0, 0.01), (0.0, 0.0, 0.41)]
    shelf_geometry = ShelfGeometry(shelf_positions=shelf_positions)
    
    test_pos = (0.3, 0.2, 0.5)
    margin = 0.1
    
    print(f"Testing position: {test_pos}")
    print(f"Margin: {margin}")
    print(f"Shelf positions: {shelf_positions}")
    print(f"Shelf dimensions: {shelf_geometry.shelf_width}x{shelf_geometry.shelf_depth}x{shelf_geometry.shelf_thickness}")
    print()
    
    for i, shelf_pos in enumerate(shelf_positions):
        x_shelf, y_shelf, z_shelf = shelf_pos
        x, y, z = test_pos
        
        within_shelf_xy = (abs(x - x_shelf) <= shelf_geometry.shelf_width / 2 + margin and
                          abs(y - y_shelf) <= shelf_geometry.shelf_depth / 2 + margin)
        
        shelf_top = z_shelf + shelf_geometry.shelf_thickness / 2
        shelf_bottom = z_shelf - shelf_geometry.shelf_thickness / 2
        
        print(f"Shelf {i}: {shelf_pos}")
        print(f"  Within XY bounds: {within_shelf_xy}")
        print(f"  Shelf bottom: {shelf_bottom:.3f}")
        print(f"  Shelf top: {shelf_top:.3f}")
        print(f"  Danger zone: {shelf_bottom - margin:.3f} to {shelf_top + margin:.3f}")
        print(f"  Position Z: {z:.3f}")
        print(f"  In danger zone: {shelf_bottom - margin <= z <= shelf_top + margin}")
        print()
    
    result = shelf_geometry._check_position_clearance(test_pos, margin)
    print(f"Final result: {result}")

if __name__ == "__main__":
    debug_clearance()
#!/usr/bin/env python3
"""
Object Library - Performance-Optimized Loading System
Milestone 4: Performance-Optimized Object Library
"""

import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import pybullet as p


@dataclass
class ObjectMetadata:
    """Object information for efficient loading"""
    object_id: str
    name: str
    file_format: str  # 'ycb' or 'gso'
    mesh_file: str
    xml_file: str
    category: str
    estimated_size: float  # rough diameter for sorting
    

class ObjectLibrary:
    """Progressive object loading system optimized for M4 Pro"""
    
    def __init__(self, assets_path: str):
        self.assets_path = Path(assets_path)
        self.ycb_path = self.assets_path / "ycb"
        self.gso_path = self.assets_path / "gso"
        self.metadata: Dict[str, ObjectMetadata] = {}
        self.loaded_objects: Dict[str, int] = {}  # object_id -> pybullet_id
        self.physics_client = None
        
    def scan_objects(self) -> None:
        """Scan and catalog all available objects"""
        self._scan_ycb_objects()
        self._scan_gso_objects()
        
    def _scan_ycb_objects(self) -> None:
        """Scan YCB objects (XML+STL format)"""
        if not self.ycb_path.exists():
            return
            
        for obj_dir in self.ycb_path.iterdir():
            if not obj_dir.is_dir():
                continue
                
            # Look for XML file in tsdf subdirectory
            tsdf_dir = obj_dir / "tsdf"
            if not tsdf_dir.exists():
                continue
                
            xml_file = tsdf_dir / "nontextured.xml"
            stl_file = tsdf_dir / "nontextured.stl"
            
            if xml_file.exists() and stl_file.exists():
                # Extract category from directory name (e.g., "025_mug" -> "mug")
                parts = obj_dir.name.split('_', 1)
                category = parts[1] if len(parts) > 1 else "unknown"
                
                # Estimate size from STL file size (rough approximation)
                file_size_kb = stl_file.stat().st_size / 1024
                estimated_size = min(0.3, max(0.05, file_size_kb / 1000))  # 5cm to 30cm
                
                metadata = ObjectMetadata(
                    object_id=f"ycb_{obj_dir.name}",
                    name=obj_dir.name,
                    file_format="ycb",
                    mesh_file=str(stl_file),
                    xml_file=str(xml_file),
                    category=category,
                    estimated_size=estimated_size
                )
                self.metadata[metadata.object_id] = metadata
                
    def _scan_gso_objects(self) -> None:
        """Scan GSO objects (XML+OBJ format)"""
        if not self.gso_path.exists():
            return
            
        for obj_dir in self.gso_path.iterdir():
            if not obj_dir.is_dir():
                continue
                
            xml_file = obj_dir / "model.xml"
            obj_file = obj_dir / "model.obj"
            
            if xml_file.exists() and obj_file.exists():
                # Extract category from directory name
                category = self._extract_gso_category(obj_dir.name)
                
                # Estimate size from OBJ file size
                file_size_kb = obj_file.stat().st_size / 1024
                estimated_size = min(0.4, max(0.05, file_size_kb / 500))  # 5cm to 40cm
                
                metadata = ObjectMetadata(
                    object_id=f"gso_{obj_dir.name}",
                    name=obj_dir.name,
                    file_format="gso",
                    mesh_file=str(obj_file),
                    xml_file=str(xml_file),
                    category=category,
                    estimated_size=estimated_size
                )
                self.metadata[metadata.object_id] = metadata
                
    def _extract_gso_category(self, name: str) -> str:
        """Extract category from GSO object name"""
        name_lower = name.lower()
        if "cup" in name_lower or "mug" in name_lower:
            return "cup"
        elif "bottle" in name_lower:
            return "bottle"
        elif "bowl" in name_lower:
            return "bowl"
        elif "can" in name_lower:
            return "can"
        else:
            return "household"
            
    def get_progressive_sets(self) -> Tuple[List[str], List[str], List[str]]:
        """Get object sets for progressive loading (5, 20, 54)"""
        # Sort by size for consistent selection
        sorted_objects = sorted(self.metadata.values(), key=lambda x: x.estimated_size)
        
        # Select diverse objects for each set
        set_5 = []
        set_20 = []
        all_objects = []
        
        # Set 5: Small diverse set
        categories_5 = ["mug", "cup", "bowl", "bottle", "can"]
        for cat in categories_5:
            candidates = [obj for obj in sorted_objects if obj.category == cat]
            if candidates:
                set_5.append(candidates[0].object_id)
                
        # Fallback if categories not found
        while len(set_5) < 5 and len(sorted_objects) > len(set_5):
            obj = sorted_objects[len(set_5)]
            if obj.object_id not in set_5:
                set_5.append(obj.object_id)
                
        # Set 20: Include variety of sizes and categories
        set_20.extend(set_5)
        remaining = [obj for obj in sorted_objects if obj.object_id not in set_20]
        set_20.extend([obj.object_id for obj in remaining[:15]])
        
        # All objects
        all_objects = [obj.object_id for obj in sorted_objects]
        
        return set_5, set_20, all_objects
        
    def load_objects(self, object_ids: List[str], physics_client=None) -> bool:
        """Load specified objects into PyBullet"""
        if physics_client is not None:
            self.physics_client = physics_client
            
        if self.physics_client is None:
            return False
            
        success_count = 0
        
        for obj_id in object_ids:
            if obj_id in self.loaded_objects:
                continue
                
            metadata = self.metadata.get(obj_id)
            if not metadata:
                continue
                
            try:
                # Load mesh based on format
                if metadata.file_format == "ycb":
                    bullet_id = self._load_ycb_object(metadata)
                else:  # gso
                    bullet_id = self._load_gso_object(metadata)
                
                if bullet_id >= 0:
                    self.loaded_objects[obj_id] = bullet_id
                    success_count += 1
                    
            except Exception:
                continue
                
        return success_count > 0
        
    def _load_ycb_object(self, metadata: ObjectMetadata) -> int:
        """Load YCB object using STL mesh"""
        try:
            collision_shape = p.createCollisionShape(
                p.GEOM_MESH,
                fileName=metadata.mesh_file,
                meshScale=[1, 1, 1]
            )
            
            visual_shape = p.createVisualShape(
                p.GEOM_MESH,
                fileName=metadata.mesh_file,
                meshScale=[1, 1, 1]
            )
            
            body_id = p.createMultiBody(
                baseMass=0.1,
                baseCollisionShapeIndex=collision_shape,
                baseVisualShapeIndex=visual_shape,
                basePosition=[0, 0, 1]
            )
            
            return body_id
        except:
            # Fallback to simple shape
            return p.loadURDF("cube.urdf", [0, 0, 1])
        
    def _load_gso_object(self, metadata: ObjectMetadata) -> int:
        """Load GSO object using OBJ mesh"""
        try:
            collision_shape = p.createCollisionShape(
                p.GEOM_MESH,
                fileName=metadata.mesh_file,
                meshScale=[0.01, 0.01, 0.01]
            )
            
            visual_shape = p.createVisualShape(
                p.GEOM_MESH,
                fileName=metadata.mesh_file,
                meshScale=[0.01, 0.01, 0.01]
            )
            
            body_id = p.createMultiBody(
                baseMass=0.1,
                baseCollisionShapeIndex=collision_shape,
                baseVisualShapeIndex=visual_shape,
                basePosition=[0, 0, 1]
            )
            
            return body_id
        except:
            # Fallback to simple shape
            return p.loadURDF("cube.urdf", [0, 0, 1])
        
    def get_object_count(self) -> int:
        """Get total number of available objects"""
        return len(self.metadata)
        
    def get_loaded_count(self) -> int:
        """Get number of currently loaded objects"""
        return len(self.loaded_objects)
        
    def unload_all(self) -> None:
        """Remove all loaded objects from simulation"""
        for bullet_id in self.loaded_objects.values():
            try:
                p.removeBody(bullet_id)
            except:
                pass  # Body may already be removed
        self.loaded_objects.clear()
        
    def get_metadata(self, object_id: str) -> Optional[ObjectMetadata]:
        """Get metadata for specific object"""
        return self.metadata.get(object_id)
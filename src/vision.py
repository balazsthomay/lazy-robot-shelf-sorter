#!/usr/bin/env python3
"""
Vision System Architecture
Phase 2: DINO integration with visual similarity matching
"""

import os
import pickle
import hashlib
import numpy as np
import pybullet as p
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
from interfaces import SimulationComponent

# DINO model imports
try:
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoImageProcessor
    from PIL import Image
    DINO_AVAILABLE = True
except ImportError:
    DINO_AVAILABLE = False
    torch = None


@dataclass
class CameraConfiguration:
    """Camera system configuration parameters"""
    image_width: int = 640
    image_height: int = 480
    fov: float = 60.0  # Field of view in degrees
    near_plane: float = 0.02  # Tightened for better depth accuracy
    far_plane: float = 5.0    # Reduced from 10.0


@dataclass
class RGBDData:
    """RGB-D camera capture data"""
    rgb_image: np.ndarray
    depth_image: np.ndarray
    camera_position: Tuple[float, float, float]
    camera_target: Tuple[float, float, float]
    timestamp: float


class Camera:
    """Individual camera with capture capabilities"""
    
    def __init__(self, position: Tuple[float, float, float], 
                 target: Tuple[float, float, float],
                 config: CameraConfiguration,
                 physics_client: int = None):
        self.position = position
        self.target = target
        self.config = config
        self.physics_client = physics_client
        self.view_matrix = None
        self.projection_matrix = None
        self._update_matrices()
        
    def _update_matrices(self) -> None:
        """Update view and projection matrices"""
        # View matrix
        self.view_matrix = p.computeViewMatrix(
            cameraEyePosition=self.position,
            cameraTargetPosition=self.target,
            cameraUpVector=[0, 0, 1]
        )
        
        # Projection matrix
        aspect = self.config.image_width / self.config.image_height
        self.projection_matrix = p.computeProjectionMatrixFOV(
            fov=self.config.fov,
            aspect=aspect,
            nearVal=self.config.near_plane,
            farVal=self.config.far_plane
        )
        
    def capture_rgbd(self) -> RGBDData:
        """Capture RGB-D image"""
        # Capture image using PyBullet
        width, height, rgb_img, depth_img, _ = p.getCameraImage(
            width=self.config.image_width,
            height=self.config.image_height,
            viewMatrix=self.view_matrix,
            projectionMatrix=self.projection_matrix,
            physicsClientId=self.physics_client
        )
        
        # Convert RGB to numpy array (remove alpha channel)
        rgb_array = np.array(rgb_img).reshape(height, width, 4)[:, :, :3]
        
        # Convert depth to numpy array and normalize
        depth_array = np.array(depth_img).reshape(height, width)
        far = self.config.far_plane
        near = self.config.near_plane
        depth_normalized = far * near / (far - (far - near) * depth_array)
        
        import time
        return RGBDData(
            rgb_image=rgb_array,
            depth_image=depth_normalized,
            camera_position=self.position,
            camera_target=self.target,
            timestamp=time.time()
        )


class CameraSystem(SimulationComponent):
    """Flexible camera system supporting multiple configurations"""
    
    def __init__(self, config: CameraConfiguration, physics_client: int = None):
        self.config = config
        self.cameras: Dict[str, Camera] = {}
        self.physics_client = physics_client
        
    def initialize(self, use_gui: bool = False) -> None:
        """Initialize camera system"""
        # Camera system doesn't need physics client setup
        pass
        
    def setup_front_facing_camera(self, robot_height: float = 0.8) -> Camera:
        """Primary front-facing camera at robot height"""
        # Position camera in front of shelf area
        camera_position = (1.0, 0.0, robot_height)  # 1m in front of shelf
        camera_target = (0.0, 0.0, robot_height * 0.7)  # Look at shelf area
        
        camera = Camera(camera_position, camera_target, self.config, self.physics_client)
        self.cameras["front"] = camera
        return camera
        
    def setup_top_down_camera(self, shelf_center: Tuple[float, float, float]) -> Camera:
        """Angled overhead camera for object detection (avoids PyBullet vertical rendering issues)"""
        x, y, z = shelf_center
        
        # Position camera at working angle that avoids PyBullet's vertical rendering bug
        # Based on successful debug test: angled cameras work, vertical ones fail
        camera_position = (x + 0.15, y - 0.15, z + 0.7)  # Smaller offset, closer camera
        camera_target = (x, y, z)  # Look at target center
        
        camera = Camera(camera_position, camera_target, self.config, self.physics_client)
        camera_name = f"top_down_{len([k for k in self.cameras.keys() if k.startswith('top_down')])}"
        self.cameras[camera_name] = camera
        return camera
        
    def capture_all_cameras(self) -> Dict[str, RGBDData]:
        """Capture RGB-D data from all cameras"""
        captures = {}
        for name, camera in self.cameras.items():
            captures[name] = camera.capture_rgbd()
        return captures
        
    def get_camera(self, name: str) -> Optional[Camera]:
        """Get camera by name"""
        return self.cameras.get(name)
        
    def cleanup(self) -> None:
        """Clean up camera resources"""
        self.cameras.clear()
        
    def get_state(self) -> dict:
        """Get current camera system state"""
        return {
            "num_cameras": len(self.cameras),
            "camera_names": list(self.cameras.keys()),
            "config": {
                "width": self.config.image_width,
                "height": self.config.image_height,
                "fov": self.config.fov
            }
        }


class DinoModel(SimulationComponent):
    """Simple DINO model for feature extraction"""
    
    def __init__(self):
        if not DINO_AVAILABLE:
            raise ImportError("Install torch and transformers for DINO support")
        self.model = None
        self.processor = None
        
    def initialize(self, use_gui: bool = False) -> None:
        """Load DINO model with improved error handling and fallback"""
        print("🔧 Initializing DINO model...")
        
        # Try DINOv3 first (primary model) - but it may not be available yet
        model_name = "facebook/dinov3-vitb16-pretrain-lvd1689m"
        try:
            print(f"📥 Attempting to load {model_name}...")
            
            # Load with specific parameters for DINOv3
            self.processor = AutoImageProcessor.from_pretrained(
                model_name,
                trust_remote_code=True
            )
            self.model = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.float32
            )
            
            print(f"✅ Successfully loaded {model_name}")
            self._current_model = model_name
            
        except Exception as e:
            print(f"⚠️  DINOv3 loading failed: {str(e)}")
            if "Unrecognized image processor" in str(e):
                print("💡 DINOv3 may not be fully supported in current transformers version")
            print("📥 Falling back to DINOv2...")
            
            # Fallback to DINOv2 (battle-tested)
            model_name = "facebook/dinov2-base"
            try:
                self.processor = AutoImageProcessor.from_pretrained(model_name)
                self.model = AutoModel.from_pretrained(
                    model_name,
                    torch_dtype=torch.float32
                )
                print(f"✅ Successfully loaded fallback {model_name}")
                self._current_model = model_name
                
            except Exception as fallback_error:
                print(f"❌ Fallback loading also failed: {str(fallback_error)}")
                raise RuntimeError(f"Could not load any DINO model. Primary error: {e}, Fallback error: {fallback_error}")
        
        self.model.eval()
        print(f"🎯 Model ready: {self._current_model}")
        
    def extract_features(self, image: Union[np.ndarray, 'Image.Image']) -> np.ndarray:
        """Extract normalized features from image"""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype(np.uint8))
            
        inputs = self.processor(image, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            features = outputs.last_hidden_state[:, 0, :].numpy().flatten()
            
        # Normalize for cosine similarity
        return features / (np.linalg.norm(features) + 1e-8)
        
    def cleanup(self) -> None:
        """Clean up resources"""
        self.model = None
        self.processor = None
        
    def get_model_info(self) -> dict:
        """Get current model information"""
        if not self.model:
            return {"initialized": False}
        
        # Use tracked model name or try to get from model config
        model_name = getattr(self, '_current_model', "unknown")
        if model_name == "unknown":
            if hasattr(self.model, 'config') and hasattr(self.model.config, '_name_or_path'):
                model_name = self.model.config._name_or_path
            elif hasattr(self.model, 'name_or_path'):
                model_name = self.model.name_or_path
        
        return {
            "initialized": True,
            "model_name": model_name,
            "embedding_dim": 768,
            "device": "cpu",
            "is_dinov3": "dinov3" in model_name.lower()
        }
        
    def get_state(self) -> dict:
        return {"initialized": self.model is not None}


class ShelfZoneManager(SimulationComponent):
    """Simple 3x2 zone manager for 3 shelves (18 total zones)"""
    
    def __init__(self):
        self.zones = {}  # zone_id -> (x, y, z) center
        
    def initialize(self, use_gui: bool = False) -> None:
        """Create 18 zones: 3 shelves × 6 zones each"""
        zone_id = 0
        for shelf in range(3):  # 3 shelves
            shelf_z = shelf * 0.4  # 40cm between shelves
            for x in range(3):  # 3 zones width
                for y in range(2):  # 2 zones height
                    center = (x * 0.3, y * 0.2, shelf_z)  # Simple grid
                    self.zones[f"zone_{zone_id}"] = center
                    zone_id += 1
        print(f"✅ Created {len(self.zones)} zones")
        
    def get_zone_center(self, zone_id: str) -> Optional[Tuple[float, float, float]]:
        """Get zone center coordinates"""
        return self.zones.get(zone_id)
        
    def get_all_zones(self) -> Dict[str, Tuple[float, float, float]]:
        """Get all zones"""
        return self.zones.copy()
        
    def cleanup(self) -> None:
        self.zones.clear()
        
    def get_state(self) -> dict:
        return {"num_zones": len(self.zones)}


class FeatureCache:
    """Simple file-based feature caching"""
    
    def __init__(self, cache_dir: str = None):
        if cache_dir is None:
            cache_dir = os.path.expanduser("~/.cache/shelf-sorter")
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
    def _get_cache_path(self, key: str) -> str:
        """Get cache file path for key"""
        safe_key = key.replace("/", "_").replace(":", "_")
        return os.path.join(self.cache_dir, f"{safe_key}.pkl")
        
    def store(self, key: str, embedding: np.ndarray) -> None:
        """Store embedding in cache"""
        cache_path = self._get_cache_path(key)
        with open(cache_path, 'wb') as f:
            pickle.dump(embedding, f)
            
    def load(self, key: str) -> Optional[np.ndarray]:
        """Load embedding from cache"""
        cache_path = self._get_cache_path(key)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except Exception:
                return None
        return None
        
    def clear(self) -> None:
        """Clear all cached embeddings"""
        if os.path.exists(self.cache_dir):
            for file in os.listdir(self.cache_dir):
                if file.endswith('.pkl'):
                    os.remove(os.path.join(self.cache_dir, file))
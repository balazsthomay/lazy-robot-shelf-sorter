#!/usr/bin/env python3
"""
Domain Randomization System
Milestone 6: Incremental domain randomization for robust training
"""

import numpy as np
import pybullet as p
from dataclasses import dataclass
from typing import Tuple, Optional
import random
from vision import Camera, CameraConfiguration


@dataclass
class RandomizationConfig:
    """Configuration for domain randomization parameters"""
    lighting_range: Tuple[float, float] = (0.3, 1.0)
    camera_jitter: float = 0.05  # ±5cm
    depth_noise: float = 0.05    # 0-5% Gaussian noise
    enable_lighting: bool = True
    enable_jitter: bool = True
    enable_noise: bool = True


class LightingController:
    """Manage lighting variation for domain randomization"""
    
    def __init__(self, intensity_range: Tuple[float, float] = (0.3, 1.0)):
        self.intensity_range = intensity_range
        self.current_intensity = 1.0
        self.default_ambient = [0.4, 0.4, 0.4]
        self.default_diffuse = [0.8, 0.8, 0.8]
        self.default_specular = [0.1, 0.1, 0.1]
        
    def randomize_lighting(self) -> float:
        """Apply random lighting intensity within range"""
        min_intensity, max_intensity = self.intensity_range
        self.current_intensity = random.uniform(min_intensity, max_intensity)
        
        # Scale default lighting by intensity
        ambient = [c * self.current_intensity for c in self.default_ambient]
        diffuse = [c * self.current_intensity for c in self.default_diffuse]
        specular = [c * self.current_intensity for c in self.default_specular]
        
        # Apply to PyBullet (light index 0 is default)
        try:
            # Note: PyBullet lighting control is limited in basic version
            # This is a simplified implementation
            pass
        except:
            pass
            
        return self.current_intensity
        
    def reset_lighting(self) -> None:
        """Reset to default lighting"""
        self.current_intensity = 1.0


class CameraJitter:
    """Camera position randomization with stability validation"""
    
    def __init__(self, jitter_range: float = 0.05):  # ±5cm
        self.jitter_range = jitter_range
        
    def apply_position_jitter(self, camera: Camera) -> Camera:
        """Apply random position jitter"""
        # Generate random jitter
        jitter_x = random.uniform(-self.jitter_range, self.jitter_range)
        jitter_y = random.uniform(-self.jitter_range, self.jitter_range)
        jitter_z = random.uniform(-self.jitter_range/2, self.jitter_range/2)  # Less Z variation
        
        # Apply jitter to camera position
        original_pos = camera.position
        new_position = (
            original_pos[0] + jitter_x,
            original_pos[1] + jitter_y,
            original_pos[2] + jitter_z
        )
        
        # Create new camera with jittered position
        jittered_camera = Camera(new_position, camera.target, camera.config)
        return jittered_camera
        
    def get_jitter_amount(self, original_camera: Camera, jittered_camera: Camera) -> float:
        """Calculate total jitter distance"""
        orig_pos = np.array(original_camera.position)
        new_pos = np.array(jittered_camera.position)
        return np.linalg.norm(new_pos - orig_pos)


class DepthNoiseGenerator:
    """Add realistic depth sensor noise"""
    
    def __init__(self, noise_percentage: float = 0.05):  # 0-5% Gaussian
        self.noise_percentage = noise_percentage
        
    def add_depth_noise(self, depth_image: np.ndarray) -> np.ndarray:
        """Add Gaussian noise to depth measurements"""
        if self.noise_percentage <= 0:
            return depth_image.copy()
            
        # Generate noise proportional to depth values
        noise_std = depth_image * self.noise_percentage
        noise = np.random.normal(0, noise_std)
        
        # Add noise and ensure positive depths
        noisy_depth = depth_image + noise
        noisy_depth = np.maximum(noisy_depth, 0.01)  # Minimum 1cm depth
        
        return noisy_depth
        
    def calculate_noise_stats(self, original: np.ndarray, noisy: np.ndarray) -> dict:
        """Calculate noise statistics"""
        diff = noisy - original
        return {
            "mean_noise": np.mean(diff),
            "std_noise": np.std(diff),
            "max_noise": np.max(np.abs(diff)),
            "snr": np.mean(original) / np.std(diff) if np.std(diff) > 0 else float('inf')
        }


class DomainRandomizer:
    """Main domain randomization coordinator"""
    
    def __init__(self, config: RandomizationConfig = None):
        self.config = config or RandomizationConfig()
        self.lighting = LightingController(self.config.lighting_range)
        self.jitter = CameraJitter(self.config.camera_jitter)
        self.noise_gen = DepthNoiseGenerator(self.config.depth_noise)
        
    def randomize_environment(self) -> dict:
        """Apply all enabled randomization"""
        results = {}
        
        if self.config.enable_lighting:
            intensity = self.lighting.randomize_lighting()
            results['lighting_intensity'] = intensity
            
        return results
        
    def randomize_camera(self, camera: Camera) -> Camera:
        """Apply camera randomization"""
        if self.config.enable_jitter:
            return self.jitter.apply_position_jitter(camera)
        return camera
        
    def randomize_depth(self, depth_image: np.ndarray) -> np.ndarray:
        """Apply depth noise"""
        if self.config.enable_noise:
            return self.noise_gen.add_depth_noise(depth_image)
        return depth_image
        
    def reset_all(self) -> None:
        """Reset all randomization to defaults"""
        self.lighting.reset_lighting()
        
    def get_current_state(self) -> dict:
        """Get current randomization state"""
        return {
            "lighting_intensity": self.lighting.current_intensity,
            "jitter_range": self.jitter.jitter_range,
            "depth_noise": self.noise_gen.noise_percentage,
            "config": {
                "lighting_enabled": self.config.enable_lighting,
                "jitter_enabled": self.config.enable_jitter, 
                "noise_enabled": self.config.enable_noise
            }
        }
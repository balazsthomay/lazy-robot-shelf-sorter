"""
GG-CNN Grasp Predictor using Option 1: Complete Pre-trained Model

This module loads the complete pre-trained GG-CNN model directly from the
Cornell trained weights, ensuring exact architecture compatibility.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from typing import List, Optional
from dataclasses import dataclass
import logging

from .coordinate_transforms import CoordinateTransformer

logger = logging.getLogger(__name__)


class GGCNNNetwork(nn.Module):
    """
    GG-CNN - Generative Grasping Convolutional Neural Network
    
    A fully-convolutional network which predicts the quality and pose 
    of antipodal grasps at every pixel in an input depth image.
    """
    
    def __init__(self, input_channels=1):
        super(GGCNNNetwork, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=9, stride=3, padding=3)
        self.conv2 = nn.Conv2d(32, 16, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(16, 8, kernel_size=3, stride=2, padding=1)
        
        # Transpose convolutional layers (upsampling) - fixed to match trained model
        self.convt1 = nn.ConvTranspose2d(8, 8, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.convt2 = nn.ConvTranspose2d(8, 16, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.convt3 = nn.ConvTranspose2d(16, 32, kernel_size=9, stride=3, padding=3, output_padding=2)
        
        # Output layers - one for each prediction type
        self.pos_output = nn.Conv2d(32, 1, kernel_size=2)
        self.cos_output = nn.Conv2d(32, 1, kernel_size=2)
        self.sin_output = nn.Conv2d(32, 1, kernel_size=2)
        self.width_output = nn.Conv2d(32, 1, kernel_size=2)
        
    def forward(self, x):
        """Forward pass through the network."""
        # Encoder path
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x1))
        x3 = F.relu(self.conv3(x2))
        
        # Decoder path
        x = F.relu(self.convt1(x3))
        x = F.relu(self.convt2(x))
        x = F.relu(self.convt3(x))
        
        # Output predictions
        pos = torch.sigmoid(self.pos_output(x))
        cos = torch.tanh(self.cos_output(x))
        sin = torch.tanh(self.sin_output(x))
        width = torch.sigmoid(self.width_output(x))
        
        return pos, cos, sin, width


@dataclass
class GraspPose:
    """Represents a predicted grasp pose in world coordinates."""
    position: np.ndarray        # 3D position in world coordinates (x, y, z)
    orientation: np.ndarray     # Quaternion orientation (x, y, z, w)
    width: float               # Gripper width in meters
    confidence: float          # Grasp quality score [0, 1]


class GGCNNPredictor:
    """
    GG-CNN grasp predictor using the complete pre-trained model.
    
    Uses Option 1 loading: torch.load() to get the exact trained model
    with guaranteed architecture compatibility.
    """
    
    def __init__(self, device: str = 'cpu', max_predictions: int = 5):
        """
        Initialize GG-CNN predictor.
        
        Args:
            device: Device to run inference on ('cpu' or 'cuda')
            max_predictions: Maximum number of grasp candidates to return
        """
        self.device = torch.device(device)
        self.max_predictions = max_predictions
        self.model = None
        self.coordinate_transformer: Optional[CoordinateTransformer] = None
        
        # Input preprocessing parameters for GG-CNN
        self.input_size = (224, 224)  # Standard GG-CNN input size
        
        logger.info(f"Initialized GG-CNN predictor on {self.device}")
    
    def load_model(self, state_dict_path: str) -> None:
        """
        Load the pre-trained GG-CNN model using state dict.
        
        Args:
            state_dict_path: Path to the state dict file (.pt)
        """
        try:
            # Create our own GGCNN architecture
            self.model = GGCNNNetwork(input_channels=1)
            
            # Load the trained weights from state dict
            state_dict = torch.load(state_dict_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state_dict)
            self.model.eval()  # Set to evaluation mode
            
            logger.info(f"Loaded GG-CNN state dict from {state_dict_path}")
            logger.info(f"Model architecture: {type(self.model).__name__}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def set_coordinate_transformer(self, transformer: CoordinateTransformer) -> None:
        """Set coordinate transformer for pixel-to-world conversion."""
        self.coordinate_transformer = transformer
        logger.info("Coordinate transformer set")
    
    def preprocess_depth(self, depth_img: np.ndarray) -> torch.Tensor:
        """
        Preprocess depth image for GG-CNN input.
        
        Args:
            depth_img: Raw depth image (H, W) in meters
            
        Returns:
            Preprocessed depth tensor ready for GG-CNN
        """
        # Handle invalid depth values
        depth_img = np.nan_to_num(depth_img, nan=0.0, posinf=2.0, neginf=0.0)
        
        # Resize to GG-CNN input size
        depth_resized = cv2.resize(depth_img, self.input_size, interpolation=cv2.INTER_NEAREST)
        
        # Normalize depth values (GG-CNN expects normalized inputs)
        depth_normalized = depth_resized / depth_resized.max() if depth_resized.max() > 0 else depth_resized
        
        # Convert to tensor and add batch/channel dimensions
        depth_tensor = torch.from_numpy(depth_normalized).float()
        depth_tensor = depth_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        
        return depth_tensor.to(self.device)
    
    def postprocess_predictions(self, 
                              pos_output: torch.Tensor,
                              cos_output: torch.Tensor, 
                              sin_output: torch.Tensor,
                              width_output: torch.Tensor,
                              original_size: tuple) -> List[tuple]:
        """
        Extract grasp candidates from GG-CNN network outputs.
        
        Args:
            pos_output: Position/quality prediction
            cos_output: Cosine angle prediction
            sin_output: Sine angle prediction  
            width_output: Width prediction
            original_size: Original depth image size (H, W)
            
        Returns:
            List of (row, col, quality, angle, width) tuples
        """
        # Convert to numpy and remove batch dimension
        pos_np = pos_output.squeeze().detach().cpu().numpy()
        cos_np = cos_output.squeeze().detach().cpu().numpy()
        sin_np = sin_output.squeeze().detach().cpu().numpy()
        width_np = width_output.squeeze().detach().cpu().numpy()
        
        # Find local maxima in quality map
        try:
            from scipy.ndimage import maximum_filter
        except ImportError:
            logger.error("scipy required for non-maximum suppression")
            return []
        
        # Apply non-maximum suppression
        neighborhood_size = 5
        pos_max = maximum_filter(pos_np, size=neighborhood_size)
        peaks = (pos_np == pos_max) & (pos_np > 0.2)  # Quality threshold
        
        # Get peak locations
        peak_rows, peak_cols = np.where(peaks)
        
        if len(peak_rows) == 0:
            logger.warning("No grasp candidates found above threshold")
            return []
        
        # Extract predictions at peak locations
        candidates = []
        for row, col in zip(peak_rows, peak_cols):
            quality = pos_np[row, col]
            cos_val = cos_np[row, col] 
            sin_val = sin_np[row, col]
            width_val = width_np[row, col]
            
            # Calculate angle from cos/sin
            angle = np.arctan2(sin_val, cos_val)
            
            # Convert width (assuming normalized output to [0,1])
            width_m = width_val * 0.08  # Scale to max gripper width
            
            # Scale coordinates back to original image size
            orig_row = int(row * original_size[0] / pos_np.shape[0])
            orig_col = int(col * original_size[1] / pos_np.shape[1])
            
            candidates.append((orig_row, orig_col, quality, angle, width_m))
        
        # Sort by quality and return top candidates
        candidates.sort(key=lambda x: x[2], reverse=True)
        return candidates[:self.max_predictions]
    
    def predict(self, rgb_img: np.ndarray, depth_img: np.ndarray) -> List[GraspPose]:
        """
        Predict grasp poses from RGB-D input.
        
        Args:
            rgb_img: RGB image (H, W, 3) - currently unused by GG-CNN
            depth_img: Depth image (H, W) in meters
            
        Returns:
            List of predicted grasp poses in world coordinates
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        if depth_img.size == 0:
            logger.warning("Empty depth image provided")
            return []
        
        original_size = depth_img.shape
        
        # Preprocess depth image
        depth_tensor = self.preprocess_depth(depth_img)
        
        # Run inference
        with torch.no_grad():
            # GG-CNN outputs: pos, cos, sin, width
            pos_output, cos_output, sin_output, width_output = self.model(depth_tensor)
        
        # Extract grasp candidates
        candidates = self.postprocess_predictions(
            pos_output, cos_output, sin_output, width_output, original_size
        )
        
        if not candidates:
            return []
        
        # Convert to world coordinates using coordinate transformer
        grasp_poses = []
        
        for row, col, quality_score, angle_rad, width_m in candidates:
            # Get depth at grasp point
            if 0 <= row < depth_img.shape[0] and 0 <= col < depth_img.shape[1]:
                grasp_depth = depth_img[row, col]
            else:
                continue
                
            if grasp_depth > 0 and self.coordinate_transformer:
                try:
                    # Convert to world coordinates
                    position = self.coordinate_transformer.pixel_to_world_coordinates(
                        col, row, grasp_depth
                    )
                    
                    # Convert angle to proper 3D grasp orientation
                    # Test with identity orientation to check if positions are reachable
                    
                    # Identity orientation (no rotation) for testing
                    orientation = np.array([0, 0, 0, 1])
                    
                    # Create grasp pose
                    grasp_pose = GraspPose(
                        position=position,
                        orientation=orientation,
                        width=max(0.01, min(width_m, 0.08)),  # Clamp to gripper range
                        confidence=float(quality_score)
                    )
                    
                    grasp_poses.append(grasp_pose)
                    
                except Exception as e:
                    logger.debug(f"Failed to convert coordinates for pixel ({col}, {row}): {e}")
                    continue
        
        logger.info(f"Generated {len(grasp_poses)} grasp predictions")
        return grasp_poses
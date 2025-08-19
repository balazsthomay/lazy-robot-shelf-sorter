#!/usr/bin/env python3
"""
Tests for Phase 2 Vision System components
"""

import unittest
import numpy as np
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock

# Import components to test
import sys
sys.path.append('..')
from vision import DinoModel, ShelfZoneManager, FeatureCache
from placement import SimilarityEngine


class TestDinoModel(unittest.TestCase):
    """Test DinoModel functionality"""
    
    def setUp(self):
        """Set up test DinoModel with mocking"""
        # Skip tests if transformers not available
        try:
            from transformers import AutoModel, AutoImageProcessor
        except ImportError:
            self.skipTest("Transformers not available")
    
    def _setup_mocked_dino(self, mock_processor, mock_model):
        """Set up test DinoModel with mocks"""
        # Mock the model and processor
        mock_model_instance = Mock()
        mock_model_instance.eval.return_value = None
        mock_model.from_pretrained.return_value = mock_model_instance
        
        mock_processor_instance = Mock()
        mock_processor.from_pretrained.return_value = mock_processor_instance
        
        dino = DinoModel()
        dino.initialize()
        return dino
    
    @patch('vision.DINO_AVAILABLE', True)
    @patch('transformers.AutoModel')
    @patch('transformers.AutoImageProcessor')
    def test_initialization(self, mock_processor, mock_model):
        """Test model initialization"""
        dino = self._setup_mocked_dino(mock_processor, mock_model)
        self.assertIsNotNone(dino.model)
        self.assertIsNotNone(dino.processor)
        self.assertTrue(dino.get_state()["initialized"])
        
    @patch('vision.DINO_AVAILABLE', True)
    @patch('transformers.AutoModel')
    @patch('transformers.AutoImageProcessor')
    @patch('vision.torch')
    @patch('vision.Image')
    def test_feature_extraction(self, mock_image, mock_torch, mock_processor, mock_model):
        """Test feature extraction"""
        dino = self._setup_mocked_dino(mock_processor, mock_model)
        
        # Mock torch operations
        mock_outputs = Mock()
        mock_outputs.last_hidden_state = Mock()
        
        # Create mock features
        mock_features = np.random.random((1, 768))
        mock_outputs.last_hidden_state.__getitem__.return_value = Mock()
        mock_outputs.last_hidden_state.__getitem__.return_value.numpy.return_value.flatten.return_value = mock_features.flatten()
        
        dino.model.return_value = mock_outputs
        dino.processor.return_value = Mock(pixel_values=Mock())
        
        # Test with numpy array
        test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        features = dino.extract_features(test_image)
        
        self.assertIsInstance(features, np.ndarray)
        self.assertEqual(features.shape, (768,))
        # Should be normalized
        self.assertAlmostEqual(np.linalg.norm(features), 1.0, places=5)
        
    @patch('vision.DINO_AVAILABLE', True)
    @patch('transformers.AutoModel')
    @patch('transformers.AutoImageProcessor')
    def test_cleanup(self, mock_processor, mock_model):
        """Test resource cleanup"""
        dino = self._setup_mocked_dino(mock_processor, mock_model)
        dino.cleanup()
        self.assertIsNone(dino.model)
        self.assertIsNone(dino.processor)


class TestShelfZoneManager(unittest.TestCase):
    """Test ShelfZoneManager functionality"""
    
    def setUp(self):
        """Set up test zone manager"""
        self.zone_manager = ShelfZoneManager()
        self.zone_manager.initialize()
    
    def test_initialization(self):
        """Test zone creation"""
        state = self.zone_manager.get_state()
        self.assertEqual(state["num_zones"], 18)  # 3 shelves × 6 zones
        
        # Verify all zones exist
        zones = self.zone_manager.get_all_zones()
        self.assertEqual(len(zones), 18)
        
        # Check zone naming
        for i in range(18):
            self.assertIn(f"zone_{i}", zones)
            
    def test_zone_centers(self):
        """Test zone center coordinates"""
        # Test first zone (bottom shelf, corner)
        center = self.zone_manager.get_zone_center("zone_0")
        self.assertIsNotNone(center)
        self.assertEqual(len(center), 3)  # x, y, z
        
        # Test last zone
        center = self.zone_manager.get_zone_center("zone_17")
        self.assertIsNotNone(center)
        
        # Test invalid zone
        center = self.zone_manager.get_zone_center("invalid_zone")
        self.assertIsNone(center)
        
    def test_cleanup(self):
        """Test cleanup functionality"""
        self.zone_manager.cleanup()
        self.assertEqual(len(self.zone_manager.zones), 0)


class TestSimilarityEngine(unittest.TestCase):
    """Test SimilarityEngine functionality"""
    
    def setUp(self):
        """Set up test similarity engine"""
        self.engine = SimilarityEngine()
        self.engine.initialize()
    
    def test_initialization(self):
        """Test engine initialization"""
        state = self.engine.get_state()
        self.assertEqual(state["num_objects"], 0)
        self.assertEqual(state["num_zones"], 0)
        
    def test_embedding_storage(self):
        """Test embedding storage and retrieval"""
        # Create test embeddings
        obj_embedding = np.random.random(768)
        obj_embedding /= np.linalg.norm(obj_embedding)
        
        zone_embedding = np.random.random(768) 
        zone_embedding /= np.linalg.norm(zone_embedding)
        
        # Store embeddings
        self.engine.store_object_embedding("obj_1", obj_embedding)
        self.engine.store_zone_embedding("zone_1", zone_embedding)
        
        # Check state
        state = self.engine.get_state()
        self.assertEqual(state["num_objects"], 1)
        self.assertEqual(state["num_zones"], 1)
        
    def test_similarity_computation(self):
        """Test cosine similarity computation"""
        # Identical embeddings should have similarity = 1.0
        embedding1 = np.array([1.0, 0.0, 0.0])
        embedding2 = np.array([1.0, 0.0, 0.0])
        similarity = self.engine.compute_similarity(embedding1, embedding2)
        self.assertAlmostEqual(similarity, 1.0, places=5)
        
        # Orthogonal embeddings should have similarity = 0.0
        embedding1 = np.array([1.0, 0.0, 0.0])
        embedding2 = np.array([0.0, 1.0, 0.0])
        similarity = self.engine.compute_similarity(embedding1, embedding2)
        self.assertAlmostEqual(similarity, 0.0, places=5)
        
    def test_best_zone_finding(self):
        """Test finding best matching zone"""
        # Add zone embeddings
        zone1_emb = np.array([1.0, 0.0, 0.0])
        zone2_emb = np.array([0.0, 1.0, 0.0])
        self.engine.store_zone_embedding("zone_1", zone1_emb)
        self.engine.store_zone_embedding("zone_2", zone2_emb)
        
        # Object similar to zone_1
        obj_emb = np.array([0.9, 0.1, 0.0])
        obj_emb /= np.linalg.norm(obj_emb)
        
        best_zone = self.engine.find_best_zone(obj_emb)
        self.assertEqual(best_zone, "zone_1")
        
    def test_similarity_heatmap(self):
        """Test heatmap generation"""
        # Add multiple zones
        self.engine.store_zone_embedding("zone_1", np.array([1.0, 0.0, 0.0]))
        self.engine.store_zone_embedding("zone_2", np.array([0.0, 1.0, 0.0]))
        self.engine.store_zone_embedding("zone_3", np.array([0.0, 0.0, 1.0]))
        
        # Get heatmap for object
        obj_emb = np.array([1.0, 0.0, 0.0])
        heatmap = self.engine.get_similarity_heatmap(obj_emb)
        
        self.assertEqual(len(heatmap), 3)
        self.assertIn("zone_1", heatmap)
        self.assertIn("zone_2", heatmap)
        self.assertIn("zone_3", heatmap)
        
        # zone_1 should have highest similarity
        self.assertGreater(heatmap["zone_1"], heatmap["zone_2"])
        self.assertGreater(heatmap["zone_1"], heatmap["zone_3"])
        
    def test_cleanup(self):
        """Test cleanup functionality"""
        self.engine.store_object_embedding("obj_1", np.random.random(768))
        self.engine.store_zone_embedding("zone_1", np.random.random(768))
        
        self.engine.cleanup()
        
        state = self.engine.get_state()
        self.assertEqual(state["num_objects"], 0)
        self.assertEqual(state["num_zones"], 0)


class TestFeatureCache(unittest.TestCase):
    """Test FeatureCache functionality"""
    
    def setUp(self):
        """Set up test cache with temporary directory"""
        self.temp_dir = tempfile.mkdtemp()
        self.cache = FeatureCache(self.temp_dir)
        
    def tearDown(self):
        """Clean up temporary directory"""
        import shutil
        shutil.rmtree(self.temp_dir)
        
    def test_store_and_load(self):
        """Test storing and loading embeddings"""
        test_embedding = np.random.random(768)
        test_key = "test_object_1"
        
        # Store embedding
        self.cache.store(test_key, test_embedding)
        
        # Load embedding
        loaded_embedding = self.cache.load(test_key)
        
        self.assertIsNotNone(loaded_embedding)
        np.testing.assert_array_almost_equal(test_embedding, loaded_embedding)
        
    def test_load_nonexistent(self):
        """Test loading non-existent key"""
        result = self.cache.load("nonexistent_key")
        self.assertIsNone(result)
        
    def test_key_sanitization(self):
        """Test key sanitization for file paths"""
        unsafe_key = "facebook/dinov3-vitb16:latest"
        test_embedding = np.random.random(768)
        
        # Should not raise exception
        self.cache.store(unsafe_key, test_embedding)
        loaded = self.cache.load(unsafe_key)
        
        self.assertIsNotNone(loaded)
        np.testing.assert_array_almost_equal(test_embedding, loaded)
        
    def test_clear_cache(self):
        """Test cache clearing"""
        # Store some embeddings
        self.cache.store("key1", np.random.random(768))
        self.cache.store("key2", np.random.random(768))
        
        # Clear cache
        self.cache.clear()
        
        # Should not be able to load
        self.assertIsNone(self.cache.load("key1"))
        self.assertIsNone(self.cache.load("key2"))


class TestIntegration(unittest.TestCase):
    """Integration tests for vision system components"""
    
    def test_end_to_end_similarity(self):
        """Test complete similarity pipeline"""
        # Create components
        zone_manager = ShelfZoneManager()
        zone_manager.initialize()
        
        similarity_engine = SimilarityEngine()
        similarity_engine.initialize()
        
        # Create mock embeddings for zones
        for i in range(18):
            zone_embedding = np.random.random(768)
            zone_embedding /= np.linalg.norm(zone_embedding)
            similarity_engine.store_zone_embedding(f"zone_{i}", zone_embedding)
            
        # Test object placement
        object_embedding = np.random.random(768)
        object_embedding /= np.linalg.norm(object_embedding)
        
        best_zone = similarity_engine.find_best_zone(object_embedding)
        heatmap = similarity_engine.get_similarity_heatmap(object_embedding)
        
        # Verify results
        self.assertIsNotNone(best_zone)
        self.assertTrue(best_zone.startswith("zone_"))
        self.assertEqual(len(heatmap), 18)
        
        # Best zone should be in heatmap
        self.assertIn(best_zone, heatmap)
        
        # Cleanup
        zone_manager.cleanup()
        similarity_engine.cleanup()


if __name__ == '__main__':
    unittest.main()
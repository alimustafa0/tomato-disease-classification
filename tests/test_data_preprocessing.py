# tests/test_data_preprocessing.py

import unittest
import os
import shutil
from unittest.mock import patch, MagicMock
from PIL import Image
import numpy as np
import sys
# Import the transforms module
from torchvision import transforms # <--- ADDED THIS LINE

# Add the 'src' directory to the Python path for importing
# This ensures that 'data_preprocessing' can be found.
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.append(src_path)

# Import functions to be tested
from data_preprocessing import (
    create_directory_structure,
    get_transforms,
    preprocess_and_split_data,
    TARGET_IMAGE_SIZE,
    TRAIN_SPLIT_RATIO,
    VAL_SPLIT_RATIO,
    RANDOM_SEED
)

class TestDataPreprocessing(unittest.TestCase):

    def setUp(self):
        """
        Set up a temporary directory for raw and processed data before each test.
        This ensures tests are isolated and don't interfere with actual data.
        """
        self.test_raw_dir = os.path.join(project_root, 'test_data_raw')
        self.test_processed_dir = os.path.join(project_root, 'test_data_processed')
        self.test_classes = ['ClassA', 'ClassB']

        # Create dummy raw data for testing
        os.makedirs(self.test_raw_dir, exist_ok=True)
        for cls in self.test_classes:
            class_path = os.path.join(self.test_raw_dir, cls)
            os.makedirs(class_path, exist_ok=True)
            # Create some dummy image files
            for i in range(10): # 10 images per class for a small test set
                dummy_image_path = os.path.join(class_path, f'image_{i}.jpg')
                # Create a simple 256x256 blank image (or smaller for faster tests)
                Image.new('RGB', (256, 256), color = 'red').save(dummy_image_path)

    def tearDown(self):
        """
        Clean up temporary directories after each test.
        """
        if os.path.exists(self.test_raw_dir):
            shutil.rmtree(self.test_raw_dir)
        if os.path.exists(self.test_processed_dir):
            shutil.rmtree(self.test_processed_dir)

    def test_create_directory_structure(self):
        """
        Test if create_directory_structure creates the correct folders.
        """
        temp_base_dir = os.path.join(self.test_processed_dir, 'temp_structure')
        create_directory_structure(temp_base_dir, self.test_classes)

        self.assertTrue(os.path.isdir(os.path.join(temp_base_dir, 'train', 'ClassA')))
        self.assertTrue(os.path.isdir(os.path.join(temp_base_dir, 'val', 'ClassB')))
        self.assertTrue(os.path.isdir(os.path.join(temp_base_dir, 'test', 'ClassA')))
        # Clean up this specific test's temporary directory
        shutil.rmtree(temp_base_dir)

    def test_get_transforms(self):
        """
        Test if get_transforms returns the expected types of transforms.
        """
        train_t, val_test_t = get_transforms(TARGET_IMAGE_SIZE, data_augmentation=True)
        self.assertIsInstance(train_t, transforms.Compose)
        self.assertIsInstance(val_test_t, transforms.Compose)

        # Check for specific augmentation in train_transform
        self.assertTrue(any(isinstance(t, transforms.RandomHorizontalFlip) for t in train_t.transforms))
        self.assertTrue(any(isinstance(t, transforms.RandomRotation) for t in train_t.transforms))
        self.assertFalse(any(isinstance(t, transforms.RandomHorizontalFlip) for t in val_test_t.transforms))


    @patch('data_preprocessing.Image.open')
    @patch('data_preprocessing.train_test_split') # Patch train_test_split
    def test_preprocess_and_split_data(self, mock_train_test_split, mock_image_open):
        """
        Test the main preprocessing and splitting logic.
        This test uses mocks to control dependencies like Image.open and train_test_split.
        """
        # Mock Image.open to return a dummy image instead of reading from disk
        mock_img_instance = MagicMock()
        mock_img_instance.convert.return_value = mock_img_instance # .convert('RGB')
        mock_img_instance.size = (256, 256)
        mock_image_open.return_value.__enter__.return_value = mock_img_instance
        # Mock the save method of the PIL Image object
        mock_img_instance.save.return_value = None

        # Configure the mock's side_effect for the two calls to train_test_split
        # train_test_split is called twice in preprocess_and_split_data
        # 1. To split (all_images, all_labels) into (X_train_val, X_test, y_train_val, y_test)
        # 2. To split (X_train_val, y_train_val) into (X_train, X_val, y_train, y_val)
        mock_train_test_split.side_effect = [
            # First call result (for the initial train_val/test split)
            (['path_tv1', 'path_tv2', 'path_tv3', 'path_tv4'], # X_train_val (simulated list of paths)
             ['path_test1', 'path_test2'],                     # X_test (simulated list of paths)
             ['ClassA', 'ClassB', 'ClassA', 'ClassB'],         # y_train_val (simulated labels)
             ['ClassA', 'ClassB']),                            # y_test (simulated labels)

            # Second call result (for the train/val split from X_train_val)
            (['path_train1', 'path_train2'],                   # X_train (simulated list of paths)
             ['path_val1', 'path_val2'],                       # X_val (simulated list of paths)
             ['ClassA', 'ClassA'],                             # y_train (simulated labels)
             ['ClassB', 'ClassB'])                             # y_val (simulated labels)
        ]

        # Set up a mock for os.listdir to control class_names and image files
        with patch('data_preprocessing.os.listdir') as mock_listdir:
            # Simulate raw_data_dir content: 'ClassA', 'ClassB' folders
            mock_listdir.side_effect = lambda path: self.test_classes if path == self.test_raw_dir else \
                                                      ([f'image_{i}.jpg' for i in range(10)] if os.path.basename(path) in self.test_classes else [])

            # Run the function with mocked dependencies
            preprocess_and_split_data(self.test_raw_dir, self.test_processed_dir,
                                      image_size=TARGET_IMAGE_SIZE,
                                      train_ratio=TRAIN_SPLIT_RATIO,
                                      val_ratio=VAL_SPLIT_RATIO,
                                      random_seed=RANDOM_SEED)

            # Assertions:
            # Check if processed directories exist
            self.assertTrue(os.path.isdir(os.path.join(self.test_processed_dir, 'train')))
            self.assertTrue(os.path.isdir(os.path.join(self.test_processed_dir, 'val')))
            self.assertTrue(os.path.isdir(os.path.join(self.test_processed_dir, 'test')))

            # Check if the split function was called with stratification for both calls
            self.assertEqual(mock_train_test_split.call_count, 2)
            self.assertTrue(mock_train_test_split.call_args_list[0].kwargs['stratify'] is not None)
            self.assertTrue(mock_train_test_split.call_args_list[1].kwargs['stratify'] is not None)

            # Verify that class subdirectories were created within each split
            for split in ['train', 'val', 'test']:
                for cls in self.test_classes:
                    self.assertTrue(os.path.isdir(os.path.join(self.test_processed_dir, split, cls)))

            # Note: We are mocking Image.save, so we can't assert on actual files being written.
            # We are verifying that the data processing logic *would* have attempted to save them
            # to the correct, newly created directories.

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

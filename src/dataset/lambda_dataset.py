import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import random
import einops
import cv2

class HandwrittenCharacterDataset(Dataset):
    def __init__(
        self, 
        image_paths: list[str], 
        labels: list[str],
        max_translation: float = 5.0,
        max_rotation: float = 10.0
    ):
        """
        Dataset for loading and preprocessing handwritten character images.

        Args:
            image_paths (list[str]): List of paths to image files.
            labels (list[str]): List of ground truth strings corresponding to images.
        """
        
        self.max_translation: float = max_translation
        self.max_rotation: float = max_rotation
        
        self.image_paths = image_paths
        self.labels = labels
        
        assert len(self.image_paths) == len(self.labels), "Number of images and labels must match."

        # Sort labels and create a mapping for one-hot encoding
        self.label_set = sorted(set(labels))
        self.label_to_index = {label: idx for idx, label in enumerate(self.label_set)}

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Load an image and its corresponding one-hot encoded label, apply transformations, and return them.

        Args:
            index (int): Index of the item to fetch.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Transformed image tensor and one-hot encoded label tensor.
        """
        # Load image as grayscale
        image_path = self.image_paths[index]
        label = self.labels[index]
        
        image = Image.open(image_path).convert("L")  # Convert to grayscale
        image_np = np.array(image, dtype=np.float32) / 255.0  # Normalize to [0, 1]

        # Apply random transformations
        image_np = self._apply_random_transformations(image_np)

        # Add channel dimension and convert to torch tensor
        image_tensor = torch.from_numpy(image_np).unsqueeze(0)  # Shape: (1, rows, cols)

        # One-hot encode the label
        label_index = self.label_to_index[label]
        label_tensor = torch.zeros(len(self.label_set), dtype=torch.float32)
        label_tensor[label_index] = 1.0

        return image_tensor, label_tensor

    def _apply_random_transformations(self, image: np.ndarray) -> np.ndarray:
        """
        Apply random transformations: translation, rotation, stretching, and noise.

        Args:
            image (np.ndarray): Input image as a numpy array.

        Returns:
            np.ndarray: Transformed image.
        """
        rows, cols = image.shape

        # Translation
          # pixels
        tx = random.randint(-self.max_translation, self.max_translation)
        ty = random.randint(-self.max_translation, self.max_translation)
        translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
        image = cv2.warpAffine(
            image, 
            translation_matrix, 
            (cols, rows), 
            borderMode=cv2.BORDER_REFLECT
        )

        # Rotation
        angle = random.uniform(
            -self.max_rotation, 
            self.max_rotation
        )
        rotation_matrix = cv2.getRotationMatrix2D(
            (cols / 2, rows / 2), 
            angle, 
            1
        )
        image = cv2.warpAffine(
            image, 
            rotation_matrix, 
            (cols, rows), 
            borderMode=cv2.BORDER_REFLECT
        )

        # Stretching
        stretch_factor_x = random.uniform(0.8, 1.2)
        stretch_factor_y = random.uniform(0.8, 1.2)
        image = cv2.resize(
            image, 
            None, 
            fx=stretch_factor_x, 
            fy=stretch_factor_y, 
            interpolation=cv2.INTER_LINEAR
        )

        # Crop or pad back to original size
        image = self._resize_to_original(
            image, 
            rows, 
            cols
        )

        # Add random noise
        noise = np.random.normal(
            0, 
            0.02, 
            image.shape
        )
        image = np.clip(image + noise, 0, 1)

        return image

    def _resize_to_original(
        self, 
        image: np.ndarray, 
        target_rows: int, 
        target_cols: int
    ) -> np.ndarray:
        """
        Resize an image to match the target dimensions, either by cropping or padding.

        Args:
            image (np.ndarray): Input image.
            target_rows (int): Target number of rows.
            target_cols (int): Target number of columns.

        Returns:
            np.ndarray: Resized image.
        """
        current_rows, current_cols = image.shape

        # Crop if larger
        if current_rows > target_rows:
            start_row = (current_rows - target_rows) // 2
            image = image[start_row:start_row + target_rows, :]
        if current_cols > target_cols:
            start_col = (current_cols - target_cols) // 2
            image = image[:, start_col:start_col + target_cols]

        # Pad if smaller
        if current_rows < target_rows:
            pad_rows = target_rows - current_rows
            pad_top = pad_rows // 2
            pad_bottom = pad_rows - pad_top
            image = np.pad(image, ((pad_top, pad_bottom), (0, 0)), mode='constant', constant_values=0)
        if current_cols < target_cols:
            pad_cols = target_cols - current_cols
            pad_left = pad_cols // 2
            pad_right = pad_cols - pad_left
            image = np.pad(image, ((0, 0), (pad_left, pad_right)), mode='constant', constant_values=0)

        return image

# Example Usage
if __name__ == "__main__":
    # Example data
    image_paths = [
        r"C:\Users\LeonBass\Documents\visual_studio_code\ocr-model-training\dataset\lambda_calc\u2e-1736774323894.png", 
        r"C:\Users\LeonBass\Documents\visual_studio_code\ocr-model-training\dataset\lambda_calc\u3bb-1736774329619.png"
    ]
    labels = ["u2e", "u3bb"]

    dataset = HandwrittenCharacterDataset(
        image_paths, 
        labels
    )

    import matplotlib.pyplot as plt 

    for image_tensor, label_tensor in dataset:
        print(f"Image Tensor Shape: {image_tensor.shape}, Label Tensor: {label_tensor}")
        plt.imshow(image_tensor.numpy().squeeze())
        plt.show()
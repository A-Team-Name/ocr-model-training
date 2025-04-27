import torch
import random
from torch.utils.data import Dataset
from torchvision.transforms import functional as torch_func
from torchvision.io import read_image
from collections import defaultdict
from typing import List, Tuple

from torchvision.transforms import functional as F
from einops import rearrange, reduce
from torchvision.transforms import GaussianBlur
import torch.nn.functional as torch_func
from torchvision.transforms.functional import rotate, affine, resize, center_crop
from torchvision.transforms import GaussianBlur


class WordImageDataset(Dataset):
    def __init__(
        self,
        words: List[str],
        file_paths: List[str],
        labels: List[str],
        all_label_classes: List[str],
        rotation_limit: float,
        translation_limit: float,
        skew_limit: float,
        zoom_change: float,
        min_zoom: float,
        image_dims: Tuple[int, int],
        threshold: float = 0.5,
        thicken_sigma: float = 1.0,
        seed: int = 42,
        pad: int = 1
    ) -> None:
        """
        Initialize the dataset for word sequences.

        Args:
            words (List[str]): List of words (strings) to generate sequences from.
            file_paths (List[str]): List of image file paths for individual characters.
            labels (List[str]): List of character labels corresponding to `file_paths`.
            all_label_classes (List[str]): All possible character classes.
            rotation_limit (float): Maximum rotation angle in degrees.
            translation_limit (float): Maximum translation fraction (0-1).
            skew_limit (float): Maximum skew angle in degrees.
            zoom_change (float): Zoom variation range (e.g., 0.2 for ±20%).
            min_zoom (float): Minimum allowed zoom scale.
            image_dims (Tuple[int, int]): Target image dimensions (height, width).
            threshold (float): Binarization threshold. Defaults to 0.5.
            thicken_sigma (float): Sigma for thickening/thinning. Defaults to 1.0.
            seed (int): Random seed for reproducibility. Defaults to 42.
            pad (int): Padding size around the image. Defaults to 1.
        """
        self.words = words
        self.all_label_classes = all_label_classes
        self.rotation_limit = rotation_limit
        self.translation_limit = translation_limit
        self.skew_limit = skew_limit
        self.zoom_change = zoom_change
        self.min_zoom = min_zoom
        self.threshold = threshold
        self.image_dims = image_dims
        self.thicken_sigma = thicken_sigma
        self.pad = pad
        self.seed = seed

        # Build mapping from character to list of image paths
        self.char_to_paths = defaultdict(list)
        for path, label in zip(file_paths, labels):
            self.char_to_paths[label].append(path)

        # Validate that all characters in words exist in the dataset
        for word in self.words:
            for char in word:
                assert char in self.char_to_paths, f"Character '{char}' not found in the dataset."

        # Label-to-index mapping
        self.label_to_index = {char: idx for idx, char in enumerate(all_label_classes)}
        self.random = random.Random(seed)

    def __len__(self) -> int:
        return len(self.words)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve a sequence of character images and their one-hot encoded labels for a word.

        Args:
            index (int): Index of the word to retrieve.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                - Transformed image tensor of shape (sequence_length, 1, H, W)
                - One-hot encoded labels of shape (sequence_length, num_classes)
        """
        word = self.words[index]
        sequence_images = []
        sequence_labels = []

        # For each character in the word, load and transform an image
        for char in word:
            # Randomly select an image path for this character
            char_paths = self.char_to_paths[char]
            selected_path = self.random.choice(char_paths)

            # Load and preprocess image
            image = read_image(selected_path).float() / 255.0  # Normalize to [0, 1]
            image = reduce(image, "c h w -> 1 h w", "max")  # Convert to grayscale

            # Apply random transformations
            image = self._apply_random_transformations(image)

            # Append to sequence
            sequence_images.append(image)

            # One-hot encode the label
            one_hot = torch.zeros(len(self.all_label_classes), dtype=torch.float32)
            one_hot[self.label_to_index[char]] = 1.0
            sequence_labels.append(one_hot)

        # Stack images and labels into tensors
        image_sequence = torch.stack(sequence_images, dim=0)  # (seq_len, 1, H, W)
        label_sequence = torch.stack(sequence_labels, dim=0)  # (seq_len, num_classes)

        return image_sequence, label_sequence

    def _apply_random_transformations(self, image: torch.Tensor) -> torch.Tensor:
        """
        Apply random transformations to a single character image (same as original).
        """
        # Pad the image to prevent cropping during transformations
        pad_size = int(max(image.shape[1], image.shape[2]) * 0.5)  # Adjust padding size as needed
        image = torch_func.pad(image, (pad_size, pad_size, pad_size, pad_size), mode='constant', value=0.0)

        # Random rotation
        rotation_angle = self.random.uniform(
            -self.rotation_limit * 365,
            self.rotation_limit * 365
        )
        image = rotate(image, rotation_angle)

        # Random translation
        max_dx = self.translation_limit * image.shape[2]
        max_dy = self.translation_limit * image.shape[1]
        dx = self.random.uniform(-max_dx, max_dx)
        dy = self.random.uniform(-max_dy, max_dy)
        image = affine(
            image,
            angle=0,
            translate=(int(dx), int(dy)),
            scale=1.0,
            shear=(0.0, 0.0)
        )

        # Random skew (shear)
        shear_x = self.random.uniform(
            -self.skew_limit * image.shape[2],
            self.skew_limit * image.shape[2]
        )
        shear_y = self.random.uniform(
            -self.skew_limit * image.shape[1],
            self.skew_limit * image.shape[1]
        )
        image = affine(
            image,
            angle=0,
            translate=(0, 0),
            scale=1.0,
            shear=(shear_x, shear_y)
        )

        # Random zoom
        zoom_scale = self.random.uniform(
            max(self.min_zoom, 1.0 - self.zoom_change),
            max(self.min_zoom, 1.0 + self.zoom_change)
        )
        zoom_height = int(image.shape[1] * zoom_scale)
        zoom_width = int(image.shape[2] * zoom_scale)

        image = resize(image, (zoom_height, zoom_width), antialias=True)
        image = center_crop(image, (image.shape[1], image.shape[2]))

        # Apply thickening or thinning using Gaussian blur with conv2d
        thicken_thinning_sigma: float = self.random.uniform(
            -self.thicken_sigma,
            self.thicken_sigma
        )
        if thicken_thinning_sigma != 0.0:
            # Determine kernel size based on sigma (make it at least 3 and odd)
            kernel_size = max(
                3,
                int(2 * round(3 * abs(thicken_thinning_sigma)) + 1)
            )
            kernel_size += (kernel_size % 2 == 0)  # Ensure kernel size is odd

            # Apply GaussianBlur
            gaussian_blur = GaussianBlur(
                kernel_size=kernel_size,
                sigma=(
                    abs(thicken_thinning_sigma),
                    abs(thicken_thinning_sigma)
                )
            )
            image = gaussian_blur(image)

            # Adjust for thickening or thinning
            if thicken_thinning_sigma > 0:
                # Thickening: Increase the character's size by lowering the threshold
                image = torch.where(
                    image > (self.threshold - thicken_thinning_sigma * 0.1),
                    torch.tensor(1.0),
                    torch.tensor(0.0)
                )
            else:
                # Thinning: Decrease the character's size by raising the threshold
                image = torch.where(
                    image > (self.threshold + abs(thicken_thinning_sigma) * 0.1),
                    torch.tensor(1.0),
                    torch.tensor(0.0)
                )

        # Threshold the image
        image = torch.where(
            image > self.threshold,
            torch.tensor(1.0),
            torch.tensor(0.0)
        )

        # Crop the image around the black pixels (foreground)
        mask = image > 0.5  # Assuming foreground is black (0) and background is white (1)
        coords = torch.nonzero(mask)
        if coords.shape[0] > 0:
            x_min, y_min = coords.min(dim=0)[0][1:].tolist()
            x_max, y_max = coords.max(dim=0)[0][1:].tolist()
            image = image[:, x_min:x_max + 1, y_min:y_max + 1]
        else:
            # If no black pixels are found, return the original image
            pass

        # Resize to the desired dimensions
        image: torch.Tensor = resize(image, [d - self.pad*2 for d in self.image_dims], antialias=True)
        image = (image > self.threshold).type(torch.uint8).type(torch.float32)

        image = torch_func.pad(
            image,
            (self.pad, self.pad, self.pad, self.pad),
            mode='constant',
            value=0.0
        )

        return image

    def get_label_mapping(self) -> dict:
        return self.label_to_index

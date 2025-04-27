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
import numpy.typing as npt
import numpy as np


class CharImageDataset(Dataset):
    def __init__(
        self,
        file_paths: list[str],
        labels: list[str],
        all_label_classes: list[str],
        rotation_limit: float,
        translation_limit: float,
        skew_limit: float,
        zoom_change: float,
        min_zoom: float,
        image_dims: tuple[int, int],
        threshold: float = 0.5,
        thicken_sigma: float = 1.0,
        seed: int = 42,
        pad: int = 1
    ) -> None:
        """
        Initialize the dataset.

        Args:
            file_paths (list[str]): list of image file paths.
            labels (list[str]): list of string labels corresponding to the file paths.
            rotation_limit (float): Maximum rotation angle in degrees.
            translation_limit (float): Maximum translation fraction (0-1).
            skew_limit (float): Maximum skew angle in degrees.
            seed (int): Random seed for transformations. Defaults to 42.
        """
        assert len(file_paths) == len(labels), "file_paths and labels must have the same length."

        self.file_paths: list[str] = file_paths
        self.labels: list[str] = labels
        _labels_set = set([])
        self.labels_set: list[str] = []

        for label in all_label_classes:
            if label not in _labels_set:
                _labels_set.add(label)
                self.labels_set.append(label)

        self.label_to_index = {
            label: index
            for index, label in
            enumerate(self.labels_set)
        }
        self.index_to_label = {
            index: label
            for index, label in
            enumerate(self.labels)
        }

        self.pad = pad
        self.rotation_limit = rotation_limit
        self.translation_limit = translation_limit
        self.skew_limit = skew_limit
        self.zoom_change = zoom_change
        self.min_zoom = min_zoom
        self.threshold = threshold
        self.image_dims = image_dims
        self.thicken_sigma = thicken_sigma
        self.all_label_classes = all_label_classes

        self.random = random.Random(seed)

        _images: list[npt.NDArray[np.float32]] = []
        _labels: list[npt.NDArray[np.float32]] = []

        for i, file_path in enumerate(self.file_paths):
            file_path = self.file_paths[i]
            label = self.label_to_index[self.index_to_label[i]]

            # Load and preprocess image
            image = read_image(
                file_path
            ).float() / 255.0  # Normalize to [0, 1]

            if image[0, 0, 0] > 0.001:
                image = 1.0 - image

            image = reduce(image, "c h w -> 1 h w", "max")

            # One-hot encode the label
            one_hot_label = torch.zeros(
                len(self.labels_set),
                dtype=torch.float32
            )
            one_hot_label[label] = 1.0

            _images.append(image)
            _labels.append(one_hot_label)

        self.dataset_images: list[npt.NDArray[np.float32]] = _images
        self.dataset_labels: list[npt.NDArray[np.float32]] = _labels

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(
        self,
        index: int
    ) -> tuple[
        torch.Tensor,
        torch.Tensor
    ]:
        """
        Retrieve an image and its corresponding one-hot encoded label.

        Args:
            index (int): Index of the item to retrieve.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the 
                transformed image tensor and one-hot encoded label tensor.
        """
        image: npt.NDArray[np.float32] = self.dataset_images[index]
        one_hot_label: npt.NDArray[np.float32] = self.dataset_labels[index]

        image = self._apply_random_transformations(
            image
        )

        return image, one_hot_label

    def _apply_random_transformations(
        self,
        image: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply random transformations to the image.

        Args:
            image (torch.Tensor): The input image tensor.

        Returns:
            torch.Tensor: The transformed image tensor.
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

    def get_label_mapping(self) -> dict[str, int]:
        """
        Retrieve the label-to-index mapping.

        Returns:
            dict[str, int]: A dictionary mapping labels to their indices.
        """
        return self.label_to_index

    def get_index_mapping(self) -> dict[int, str]:
        """
        Retrieve the index-to-label mapping.

        Returns:
            dict[int, str]: A dictionary mapping indices to their labels.
        """
        return self.index_to_label

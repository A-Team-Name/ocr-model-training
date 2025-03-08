
import os
from typing import Callable, Optional
import PIL
import numpy as np
import cv2
import random
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

import numpy as np
from scipy import ndimage

def rotates_image(
    image: np.ndarray,
    angles: tuple[float],
    axes: tuple[tuple[int, int]]
) -> np.ndarray:
    """
    Rotate the image by given angles.

    Args:
    - image: np.ndarray
    - angles: tuple[float], rotation angles in degrees
    - axes: tuple[tuple[int, int]], axes to rotate around

    Returns:
    - rotated_image: np.ndarray
    """
    assert len(angles) == len(axes), \
        f"angles size ({len(angles)}) must be same length as axes length ({len(axes)})"

    
    rotated_image = image.copy()
    
    angle: float
    for angle in angles:
        rotated_image = ndimage.rotate(
            image, 
            angle, 
            reshape=True,
            axes=axes,
            order=0
        )   
        
    return rotated_image
    

def rotate_image(
    image: np.ndarray, 
    angle: float, 
    axes: tuple[int, int]
) -> np.ndarray:
    """
    Rotate the image by a given angle.

    Args:
    - image: np.ndarray
    - angle: float, rotation angle in degrees
    - axes: tuple[int, int], axes to rotate

    Returns:
    - rotated_image: np.ndarray
    """
    rotated_image = ndimage.rotate(
        image, 
        angle, 
        reshape=True,
        axes=axes
    )
    return rotated_image

def translate_image(
    image: np.ndarray,
    offsets: tuple[int],
    axes: tuple[int],
) -> np.ndarray:
    """
    Translate the image by a given x and y offset.

    Args:
    - image: np.ndarray
    - offsets: tuple[int], offset in pixels for each axis
    - axes: tuple[int], index for each of the axis to offset

    Returns:
    - translated_image: np.ndarray
    """
    assert len(offsets) == len(axes), \
        f"offsets size ({len(offsets)}) must be same length as axes length ({len(axes)})"

    shift_offsets: tuple[int] = [0 for _ in range(len(image.shape))]
    for off, axis in zip(offsets, axes):
        shift_offsets[axis] = off
    
    translated_image = ndimage.shift(
        input=image, 
        shift=shift_offsets
    )
    return translated_image

def scale_image(
    image: np.ndarray,
    scaling: tuple[float],
    axes: tuple[int]
) -> np.ndarray:
    """
    Scale the image by a given float in axes.

    Args:
    - image: np.ndarray
    - scaling: tuple[float], scaling for each axis
    - axes: tuple[int], index for each of the axis to scale

    Returns:
    - translated_image: np.ndarray
    """
    assert len(scaling) == len(axes), \
        f"scaling size ({len(scaling)}) must be same length as axes length ({len(axes)})"

    scalings: list[float] = [1.0 for _ in range(len(image.shape))]
    for scale, axis in zip(scaling, axes):
        scalings[axis] = scale
    
    scaled_image: np.ndarray = ndimage.zoom(
        input=image, 
        zoom=scalings
    )
    
    return scaled_image

def cut_box_image(
    image: np.ndarray,
    slices: tuple[tuple[int, int]],
    axes: tuple[int, int]
) -> np.ndarray:
    """
    Cuts a box-shaped region from an image and sets its values to zero.

    Args:
    image (np.ndarray): The input image.
    slices (tuple[tuple[int, int]]): A tuple of tuples, where each inner tuple contains the start and stop indices of the box-shaped region to be cut.
    axes (tuple[int, int]): A tuple of two axis indices that define the plane of the box-shaped region.

    Returns:
    np.ndarray: The image with the box-shaped region set to zero.

    Notes:
    The box-shaped region is defined by the slices and axes parameters. The slices parameter specifies the start and stop indices of the region, and the axes parameter specifies the plane in which the region lies.
    """
    assert len(slices) == len(axes), \
        f"slices size ({len(slices)}) must be same length as axes length ({len(axes)})"

    indexing: list[int] = [slice(0, size) for size in image.shape]
    
    for axis_slice, axis in zip(slices, axes):
        indexing[axis] = slice(*axis_slice)
    
    cut_image: np.ndarray = np.copy(image)
    cut_image[tuple(indexing)] = 0
    
    return cut_image

def pad_image(
    image: np.ndarray,
    paddings: tuple[tuple[int, int]],
    axes: tuple[int]
) -> np.ndarray:
    
    """
    Pads an image with edge replication along specified axes.

    Args:
    image (np.ndarray): The input image.
    paddings (tuple[tuple[int, int]]): A tuple of tuples, where each inner tuple contains the padding values for the corresponding axis.
    axes (tuple[int]): A tuple of axis indices along which to pad the image.

    Returns:
    np.ndarray: The padded image.

    Notes:
    The padding values are replicated from the edge pixels of the original image. This means that the padding values will be taken from the edge pixels of the original image, rather than being set to a constant value (like zeros).
    """
    assert len(paddings) == len(axes), \
        f"paddings size ({len(paddings)}) must be same length as axes length ({len(axes)})"

    padding_values = [(0, 0)] * len(image.shape)
    for padding, axis in zip(paddings, axes):
        padding_values[axis] = padding
    image = np.pad(
        image, 
        padding_values, 
        mode='edge'
    )
    return image

def resize_image_to_shape(
    image: np.ndarray,
    new_shape: tuple[int],
    axes: tuple[int]
) -> np.ndarray:
    """
    Pads or crops an image to a specified shape along specific axes.

    Args:
    image (np.ndarray): The input image.
    new_shape (tuple[int]): The desired shape of the output image.
    axes (tuple[int]): The axes along which to pad or crop.

    Returns:
    np.ndarray: The padded or cropped image.
    """
    
    assert len(new_shape) == len(axes), \
        f"new_shape size ({len(new_shape)}) must be same length as axes length ({len(axes)})"

    
    for axis, new_size in zip(axes, new_shape):
        current_size = image.shape[axis]
        if new_size > current_size:
            # Pad the image
            pad_width: int = new_size - current_size
            left_pad: int = pad_width // 2
            right_pad: int = pad_width - left_pad
            image = pad_image(
                image, 
                ((left_pad, right_pad),), 
                (axis,)
            )
        elif new_size < current_size:
            # Crop the image
            center = current_size // 2
            image = crop_image(
                image, 
                (center,), 
                (new_size,), 
                (axis,)
            )
    return image    
  
def crop_image(
    image: np.ndarray,
    centers: tuple[int],
    widths: tuple[int],
    axes: tuple[int]
) -> np.ndarray:
    
    """
    Crops an image along specified axes.

    Args:
    image (np.ndarray): The input image.
    centers (tuple[int]): A tuple of center coordinates for the crop regions.
    widths (tuple[int]): A tuple of widths for the crop regions.
    axes (tuple[int]): A tuple of axis indices along which to crop the image.

    Returns:
    np.ndarray: The cropped image.

    Notes:
    The crop regions are defined by the centers and widths parameters. The centers parameter specifies the center coordinates of the crop regions, and the widths parameter specifies the widths of the crop regions. The axes parameter specifies the axes along which to crop the image.
    """
    
    assert len(centers) == len(widths) == len(axes), \
        f"len(centers) == len(widths) == len(axes). Found {len(centers)}, {widths}, {axes}"

    
    slices: list[slice] = [
        slice(0, size) 
        for size in image.shape
    ]
    
    for center, width, axis in zip(
        centers, 
        widths, 
        axes
    ):
        left_size: int = width//2
        right_size: int = width - left_size
        left: int = center - left_size
        right: int = center + right_size
        
        left = max(left, 0)
        right = min(right, image.shape[axis])
        
        slices[axis] = slice(left, right)
    
    return image[tuple(slices)]   

def gaussian_noise_image(
    image: np.ndarray,
    mean: float = 0.0,  # mean of the Gaussian distribution
    std_dev: float = 1.0,  # standard deviation of the Gaussian distribution
) -> np.ndarray:
    """
    Adds Gaussian noise to an image.

    Args:
    - image: The input image.
    - mean: The mean of the Gaussian distribution.
    - std_dev: The standard deviation of the Gaussian distribution.
    
    Returns:
    - The image with Gaussian noise added.
    """

    noise = np.random.normal(
        mean, 
        std_dev, 
        size=image.shape
    )

    noisy_image: np.ndarray = image + noise

    return noisy_image


def smooth(
    image: np.ndarray, 
    sigma: float, 
    k: int
) -> np.ndarray:
    """
    Applies Gaussian smoothing k times and then thresholds the image.
    
    Parameters:
    - image (np.ndarray): The input image in range [0, 255].
    - sigma (float): The standard deviation for Gaussian kernel.
    - k (int): The number of times to apply Gaussian smoothing.
    - threshold (int): The threshold value. Pixels below this will be set to 0, above to 255.
    
    Returns:
    - np.ndarray: The processed image with smoothing and threshold applied.
    """
    # Convert image to float for more precise calculations
    smoothed_image = image.astype(np.float64)
    
    # Apply Gaussian smoothing k times
    for _ in range(k):
        smoothed_image = gaussian_filter(smoothed_image, sigma=sigma)
    
    return smoothed_image

def crop_to_content(
    image: np.ndarray,
    padding: int = 0,
    pixel_value: int = 1
) -> np.ndarray:
    """
    Crop a binarized image to a centered n x n bounding box with optional padding.

    Args:
        image: A 2D numpy ndarray representing the binarized image.
        n: The target width and height for the square bounding box.
        padding: The amount of padding to add around the bounding box.

    Returns:
        A 2D numpy ndarray representing the cropped and padded image.
    """
    # Find the indices of non-zero pixels (i.e., the contents)
    rows, cols = np.where(image == pixel_value)

    # Check if there are any content pixels; if not, return an empty padded box
    if rows.size == 0 or cols.size == 0:
        return np.zeros((padding, padding), dtype=image.dtype)

    # Calculate the bounding box of the contents
    min_row, max_row = np.min(rows), np.max(rows)
    min_col, max_col = np.min(cols), np.max(cols)
    
    # Calculate the content's height and width
    content_height = max_row - min_row + 1
    content_width = max_col - min_col + 1

    # Determine the side length of the square bounding box (maximum of content dimensions or n)
    side_length = max(content_height, content_width)
    
    # Calculate the center of the original bounding box
    center_row = (min_row + max_row) // 2
    center_col = (min_col + max_col) // 2

    # Calculate the new bounds for a centered square crop of `side_length`
    start_row = max(center_row - side_length // 2, 0)
    start_col = max(center_col - side_length // 2, 0)
    end_row = min(start_row + side_length, image.shape[0])
    end_col = min(start_col + side_length, image.shape[1])

    # Crop the image to the centered square bounding box
    cropped_image = image[start_row:end_row, start_col:end_col]

    # Create a new canvas with the desired size, adding padding
    output_size = side_length + padding
    padded_image = np.zeros((output_size, output_size), dtype=image.dtype)

    # Calculate where to place the cropped image on the padded canvas
    paste_row = (output_size - cropped_image.shape[0]) // 2
    paste_col = (output_size - cropped_image.shape[1]) // 2

    # Paste the cropped image into the center of the padded canvas
    padded_image[paste_row:paste_row + cropped_image.shape[0],
                 paste_col:paste_col + cropped_image.shape[1]] = cropped_image

    return padded_image


class FixedTransform:
    
    def __init__(
        self,
        axes: tuple[int],
        rotations: tuple[float],
        translations: tuple[int],
        scales: tuple[float],
        resize_to_shape: tuple[int]
    ):
        
        assert len(axes) == len(rotations)+1 == len(translations), \
            "len(axes) == len(rotations)+1 == len(translations)"
        
        self.axes: tuple[int] = axes
        self.rotations: tuple[float] = rotations
        self.translations: tuple[int] = translations
        self.scales: tuple[float] = scales
        self.resize_to_shape: tuple[int] = resize_to_shape  
        self.rotation_axes: tuple[tuple[int, int]] = tuple(
            [
                (self.axes[i], self.axes[i+1]) 
                for i in range(len(self.rotations))
            ]
        )
    
    
    def forward(
        self,
        image: np.ndarray,
        buffer_scale: float = 2.0
    ) -> np.ndarray:

        buffer_scale = max(1.0, buffer_scale)
    
        buffer_shape: list[int] = [
            int(s * buffer_scale) 
            for s in self.resize_to_shape
        ]
    
        image = resize_image_to_shape(
            image=image,
            new_shape=buffer_shape,
            axes=self.axes
        )
    
        for rotation, axes_pair in zip(
            self.rotations, 
            self.rotation_axes
        ):
            mod_axes: list[int] = [
                a % len(image.shape) 
                for a in axes_pair
            ]  
            
            image = rotate_image(
                image=image,
                angle=rotation,
                axes=mod_axes
            )
            
        image = translate_image(
            image=image,
            offsets=self.translations,
            axes=self.axes
        )
            
        image = scale_image(
            image=image,
            scaling=self.scales,
            axes=self.axes
        )
        
        image = resize_image_to_shape(
            image=image,
            new_shape=self.resize_to_shape,
            axes=self.axes
        )
        
        return image

    def __str__(self, include_hash: bool = False) -> str:
        self_hash_string = hash(self) if include_hash else ""
        
        string: str = f"FixedTransform@{self_hash_string}(\n"
        string = f"{string}\tAxes({self.axes})\n"
        string = f"{string}\tRotation({self.rotations})\n"
        string = f"{string}\tTranslation({self.translations})\n"
        string = f"{string}\tScales({self.scales})\n"
        string = f"{string}\tResize({self.resize_to_shape})\n"
        string = f"{string})"
        
        return string
        
    def __repr__(self) -> str:
        return self.__str__(include_hash=True)
        
        
        

class TransformGenerator:
    
    def __init__(
        self, 
        axes: tuple[int],
        rotation_ranges: tuple[tuple[float, float]],
        translate_ranges: tuple[tuple[int, int]],
        scale_ranges: tuple[tuple[float, float]],
        resize_to_shape: tuple[int]
    ):
        assert len(axes) == len(rotation_ranges)+1 == len(translate_ranges), \
            "len(axes) == len(rotation_ranges)+1 == len(translate_ranges)"
        
        self.axes: tuple[int] = axes
        self.rotation_ranges: tuple[tuple[float, float]] = rotation_ranges 
        self.translate_ranges: tuple[tuple[int, int]] = translate_ranges
        self.scale_ranges: tuple[tuple[float, float]] = scale_ranges
        self.resize_to_shape: tuple[int] = resize_to_shape
    
    def get_fixed_transform(
        self
    ) -> FixedTransform:
        x = self.rotation_ranges
        rotations: tuple[float] = tuple(
            [
                random.uniform(a, b)
                for a, b in 
                self.rotation_ranges
            ]
        )
        translations: tuple[int] = tuple(
            [
                random.randint(a, b)
                for a, b in 
                self.translate_ranges
            ]
        )
        scales: tuple[float] = tuple(
            [
                random.uniform(a, b)
                for a, b in 
                self.scale_ranges
            ]
        )
        
        return FixedTransform(
            axes=self.axes, 
            rotations=rotations,
            translations=translations,
            scales=scales,
            resize_to_shape=self.resize_to_shape
        )        
    

def main() -> None:
    
    file_dirpath: str = os.path.abspath(os.path.dirname(__file__))
    
    image_path: str = os.path.join(
        file_dirpath,
        os.pardir,
        os.pardir,
        "mri",
        "PNG",
        "941_S_1363",
        "Axial_PD-T2_TSE",
        "2007-03-12_13_02_27.0",
        "I44493",
        "25.png"
    )
    image_array: np.ndarray = cv2.imread(image_path)
    image_array = image_array.transpose((2, 0, 1))
    
    transformed_images: list[np.ndarray] = []
    
    transformed_images.append(
        image_array
    )
    
    gauss_image: np.ndarray = image_array.copy()
    
    gauss_image = (gauss_image - gauss_image.mean())/gauss_image.std()
    
    gauss_image = gaussian_noise_image(
        gauss_image,
        0,
        1,
    )
    
    gauss_image = (gauss_image - gauss_image.min()) / (gauss_image.max() - gauss_image.min())
    
    transformed_images.append(
        gauss_image
    )
    
    
    fig, ax = plt.subplots(1, len(transformed_images))   
    
    for i, image in enumerate(transformed_images):
        ax[i].imshow(image.transpose((1, 2, 0)))
    
    plt.show()
    plt.close()    
    

if __name__ == "__main__":
    main()
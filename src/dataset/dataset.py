import numpy as np
from torch.utils.data import Dataset
from transformers import TrOCRProcessor
from transformers import VisionEncoderDecoderModel
from transformers import AutoTokenizer
import torch
from PIL import Image
import os
from torch.functional import F
from torch.utils.data import Dataset
from PIL import Image, ImageOps


class HandwrittenTextDataset(Dataset):
    def __init__(
        self,
        dataset_dirpath: str,
        filenames: list[str],
        label_strings: list[str],
        processor: TrOCRProcessor,
        max_target_length: int = 128,
        pad_token_overwrite: int = -100,
        size: int = 384
    ):
        assert len(label_strings) == len(filenames)
        
        self.size: int = size
        self.dataset_dirpath: str = dataset_dirpath
        self.filenames: list[str] = filenames
        self.filepaths: list[str] = [
            os.path.join(
                dataset_dirpath,
                filename
            ) for filename in filenames
        ]  
        self.label_strings: list[str] = label_strings
        self.processor: TrOCRProcessor = processor
        self.max_target_length: int = max_target_length
        self.pad_token_overwrite: int = pad_token_overwrite
        self.pad_token: int = self.processor.tokenizer.pad_token_id

        
        #self._filecache: list[tuple[torch.Tensor, torch.Tensor] | None] 
        #self._filecache = [
        #    None 
        #    for _ in 
        #    range(len(filenames))
        #]
        
        
    def pad_label_to_shape(
        self,
        label: torch.Tensor,
        value: int
    ) -> torch.Tensor:

        if label.shape[-1] < self.max_target_length:
            pad_amount = self.max_target_length - label.shape[-1]
            # Pad on the right along the last dimension
            return F.pad(label, (0, pad_amount), value=value)
        return label
    
        
    
    def encode_label(
        self,
        label_string: str
    ) -> torch.Tensor:
        labels_tensor: torch.Tensor = torch.tensor(
            self.processor.tokenizer(
                label_string, 
                padding="max_length", 
                max_length=self.max_target_length
            ).input_ids
        )
        labels_tensor = self.pad_label_to_shape(
            labels_tensor, 
            self.pad_token
        )
        mask: torch.Tensor = labels_tensor == self.pad_token
        labels_tensor[mask] = self.pad_token_overwrite
        
        return labels_tensor
        
    def decode_label(
        self, 
        tokenised_label: torch.Tensor
    ) -> str:
        
        tokenised_label = tokenised_label.detach().clone()
        
        tokenised_label = self.pad_label_to_shape(
            tokenised_label,
            self.pad_token_overwrite
        )
        
        mask: torch.Tensor = tokenised_label == self.pad_token_overwrite
        tokenised_label[mask] = self.pad_token
        
        label_str = self.processor.decode(
            tokenised_label, 
            skip_special_tokens=False
        )
        return label_str
    
    def __len__(self) -> int:
        return len(self.filenames)
    

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get item from the dataset. Zooms the image so the x-axis matches self.size,
        then pads the y-axis to self.size before feeding it into the processor.
        
        Args:
            index (int): Index of the dataset item.
            
        Returns:
            tuple[torch.Tensor, torch.Tensor]: The processed image tensor and label tensor.
        """
        # Get file name and text
        filepath: str = self.filepaths[index]
        text_label: str = self.label_strings[index]
        
        # Load the image
        text_image: Image.Image = Image.open(filepath).convert("RGB")
        
        # Get original dimensions
        original_width, original_height = text_image.size
        
        # Compute zoom factor to resize the x-axis to self.size
        target_width = self.size
        zoom_factor = target_width / original_width
        new_height = int(original_height * zoom_factor)
        
        # Resize the image to (self.size, new_height)
        text_image = text_image.resize((target_width, new_height), Image.Resampling.LANCZOS)
        
        # Pad the y-axis to match self.size
        if new_height < self.size:
            total_padding = self.size - new_height
            pad_top = total_padding // 2
            pad_bottom = total_padding - pad_top
            padding = (0, pad_top, 0, pad_bottom)  # (left, top, right, bottom)
            text_image = ImageOps.expand(text_image, border=padding, fill=(0, 0, 0))  # Pad with black
        
        # Process the padded image
        text_image_tensor: torch.Tensor = self.processor(
            text_image, 
            return_tensors="pt"
        ).pixel_values.squeeze(0)  # Remove batch dimension added by processor
        
        # Encode the label
        labels_tensor: torch.Tensor = self.encode_label(text_label)
        
        return text_image_tensor, labels_tensor

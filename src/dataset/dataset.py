from torch.utils.data import Dataset
from transformers import TrOCRProcessor
from transformers import VisionEncoderDecoderModel
from transformers import AutoTokenizer
import torch
from PIL import Image
import os
from torch.functional import F
from torch.utils.data import Dataset


class HandwrittenTextDataset(Dataset):
    def __init__(
        self,
        dataset_dirpath: str,
        filenames: list[str],
        label_strings: list[str],
        processor: TrOCRProcessor,
        max_target_length: int = 128,
        pad_token_overwrite: int = -100
    ):
        assert len(label_strings) == len(filenames)
        

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
    
        
    def set_label_padding(
        self,
        label: torch.Tensor,
        inplace=True    
    ) -> torch.Tensor:

        if label.shape[-1] < self.max_target_length:
            pad_amount = self.max_target_length - label.shape[-1]
            # Pad on the right along the last dimension
            label = F.pad(label, (0, pad_amount), value=self.pad_token_overwrite)

        out_label_tensor: torch.Tensor = label

        if not inplace:
            out_label_tensor: torch.Tensor 
            out_label_tensor = label.detach().clone()

        mask: torch.Tensor = out_label_tensor == self.pad_token_overwrite
        out_label_tensor[mask] = self.pad_token

        return out_label_tensor
    
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
        
        mask: torch.Tensor = labels_tensor == self.pad_token_overwrite
        labels_tensor[mask] = self.pad_token
        
        return labels_tensor
        
    def decode_label(
        self, 
        tokenised_label: torch.Tensor
    ) -> str:
        
        tokenised_label = tokenised_label.detach().clone()
        
        mask: torch.Tensor = tokenised_label == self.pad_token
        tokenised_label[mask] = self.pad_token_overwrite
        
        label_str = self.processor.decode(
            tokenised_label, 
            skip_special_tokens=False
        )
        return label_str
    
    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(
        self,
        index: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
     
        # get file name + text 
        filepath: str = self.filepaths[index]
        text_label: str = self.label_strings[index]
        
        text_image: Image.Image = Image.open(filepath).convert("RGB")
        text_image_tensor: torch.Tensor = self.processor(
            text_image, 
            return_tensors="pt"
        ).pixel_values
        
        labels_tensor: torch.Tensor = self.encode_label(
            text_label
        )

        return text_image_tensor.squeeze(), labels_tensor
        

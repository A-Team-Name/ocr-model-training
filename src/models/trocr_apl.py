from typing import Any
from transformers import VisionEncoderDecoderModel
from transformers import TrOCRProcessor
from transformers import PreTrainedModel
from transformers import PretrainedConfig
from transformers import AutoTokenizer
from transformers import PreTrainedTokenizerBase
from transformers import AddedToken
import torch
from PIL import Image
from torch.functional import F
from torch import nn
import torch
from evaluate import load, EvaluationModule

class Trocr_Interface:
    
    def __init__(self):
        pass


class TrocrApl(Trocr_Interface):
    
    APL_CHARS: list[str | AddedToken]
    APL_CHARS = list(
        "∇⋄⍝⍺⍵¨¯×÷←↑→↓∆∊∘∧∨∩∪≠≡≢≤≥⊂⊃⊆⊖⊢⊣⊤⊥⌈⌊⌶⌷⎕⌸⌹⌺⌽⌿⍀⍉⍋⍎⍒⍕⍙⍟⍠⍣⍤⍥⍨⍪⍬⍱⍲⍳⍴⍷⍸○⍬⊇⍛⍢⍫√"
    )
    PAD_TOKEN_ID: int = 1
    PAD_TOKEN_OVERWRITE: int = -100
    
    def __init__(
        self,
        max_target_length: int,
        early_stopping: bool = True,
        no_repeat_ngram_size: int = 2,
        length_penalty: float = 2.0,
        apl_tokeniser_path: str | None = None,
        model_checkpoint_path: str = "microsoft/trocr-base-stage1"
    ):
        super(TrocrApl, self).__init__()

        self.max_target_length: int = max_target_length
        self.apl_tokeniser_path: str | None = apl_tokeniser_path
        self.early_stopping: bool = early_stopping
        self.no_repeat_ngram_size: int = no_repeat_ngram_size
        self.length_penalty: float = length_penalty
        self.model_checkpoint_path: str = model_checkpoint_path
        self.tokeniser: PreTrainedTokenizerBase
        
        if self.apl_tokeniser_path is None:
            
            self.tokeniser: AutoTokenizer = AutoTokenizer.from_pretrained(
                "microsoft/trocr-base-printed"
            )

            self.tokeniser.add_tokens(
                new_tokens=self.APL_CHARS
            )
        else:
            self.tokeniser: AutoTokenizer = AutoTokenizer.from_pretrained(
                self.apl_tokeniser_path
            )

        _processor: tuple[TrOCRProcessor, dict[str, Any]] | TrOCRProcessor
        _processor = TrOCRProcessor.from_pretrained(
            "microsoft/trocr-base-handwritten"
        )
        
        if isinstance(_processor, tuple):
            _processor = _processor[0]
            
        self.processor: TrOCRProcessor = _processor
        self.processor.tokenizer = self.tokeniser
        
        self.model: PreTrainedModel 
        self.model = VisionEncoderDecoderModel.from_pretrained(
            self.model_checkpoint_path
        )
        
        self.character_error_rate_metric: EvaluationModule = load("cer")
        self.model.config.decoder_start_token_id = self.tokeniser.cls_token_id
        self.model.config.pad_token_id = self.tokeniser.pad_token_id
        self.model.config.eos_token_id = self.tokeniser.sep_token_id
        self.model.config.vocab_size = len(self.tokeniser)
        self.model.decoder.resize_token_embeddings(len(self.tokeniser))

        self.model.config.max_length = self.max_target_length
        self.model.config.early_stopping = self.early_stopping
        self.model.config.no_repeat_ngram_size = self.no_repeat_ngram_size
        self.model.config.length_penalty = self.length_penalty
        self.model.config.num_beams = 5
        
    def forward(
        self,
        pixels: torch.Tensor,
        encoded_labels: torch.Tensor
    ):
        return self.model(
            pixel_values=pixels,
            labels=encoded_labels
        )
        
    
    def inference(
        self, 
        pixel_values: torch.Tensor
    ) -> str | list[str]:
        # Generate sequences from the model
        generated_ids: torch.Tensor = self.model.generate(
            pixel_values, 
            max_length=self.max_target_length
        )
        
        # Decode the generated IDs to text
        decoded_texts: str | list[str] = self.processor.batch_decode(
            generated_ids, 
            skip_special_tokens=True
        )
        
        return decoded_texts
    
    def compute_character_error_rate(
        self,
        y_hat_ids: torch.Tensor,
        y_ids: torch.Tensor
    ) -> dict[Any, Any] | None:
        
        y_hat_ids_padded: torch.Tensor = self.pad_label(
            y_hat_ids,
            self.PAD_TOKEN_OVERWRITE
        )
        y_ids_padded: torch.Tensor = self.pad_label(
            y_ids,
            self.PAD_TOKEN_OVERWRITE
        )
        
        y_mask: torch.Tensor = y_ids_padded == self.PAD_TOKEN_OVERWRITE
        y_ids_padded_fix = torch.where(y_mask, self.PAD_TOKEN_ID, y_ids_padded)
        
        y_hat_mask: torch.Tensor = y_hat_ids_padded == self.PAD_TOKEN_OVERWRITE
        y_hat_ids_padded_fix = torch.where(y_hat_mask, self.PAD_TOKEN_ID, y_hat_ids_padded)
        
        
        y_hat_str: str | list[str] = self.processor.batch_decode(
            y_hat_ids_padded_fix, 
            skip_special_tokens=True
        )
        
        y_str: str | list[str] = self.processor.batch_decode(
            y_ids_padded_fix,
            skip_special_tokens=False
        )

        cer: dict[Any, Any] | None = self.character_error_rate_metric.compute(
            predictions=y_hat_str,
            references=y_str
        )

        return cer
    
    def pad_label(
        self,
        label: torch.Tensor,
        value: int
    ) -> torch.Tensor:
        
        if label.shape[-1] < self.max_target_length:
            pad_amount = self.max_target_length - label.shape[-1]
            # Pad on the right along the last dimension
            _label = F.pad(
                label, 
                (0, pad_amount),
                value=value
            )
        return label
    
    def encode_label(
        self,
        label_string: str
    ) -> torch.Tensor:
        labels_tensor: torch.Tensor = torch.tensor(
            self.tokeniser(
                label_string, 
                padding="max_length", 
                max_length=self.max_target_length
            ).input_ids
        )
        labels_tensor = self.pad_label(
            labels_tensor, 
            self.PAD_TOKEN_ID
        )
        mask: torch.Tensor = labels_tensor == self.PAD_TOKEN_ID
        labels_tensor[mask] = self.PAD_TOKEN_OVERWRITE
        
        return labels_tensor
    
    def decode_model_output(
        self,
        encoded_label: torch.Tensor
    ) -> list[str]:
        
        _restored_label: torch.Tensor
        _restored_label = self.pad_label(
            encoded_label,
            self.PAD_TOKEN_OVERWRITE
        )
        
        if len(_restored_label.shape) == 1:
            _restored_label = _restored_label.unsqueeze(0)
        
        mask: torch.Tensor = _restored_label == self.PAD_TOKEN_OVERWRITE
        _restored_label = torch.where(mask, self.PAD_TOKEN_ID, _restored_label)
        
        strings: list[str] = []
        
        for label_tensor in _restored_label:
            label_str: str = self.processor.decode(
                label_tensor, 
                skip_special_tokens=True
            )
            strings.append(label_str)
        
        return strings
   
    
    def __decode_model_output(
        self,
        encoded_label: torch.Tensor
    ) -> list[str]:
        
        _restored_label: torch.Tensor
        _restored_label = self.pad_label(
            encoded_label,
            self.PAD_TOKEN_OVERWRITE
        )
        
        if len(_restored_label.shape) == 1:
            _restored_label = _restored_label.unsqueeze(0)
        
        mask: torch.Tensor = _restored_label == self.PAD_TOKEN_OVERWRITE
        _restored_label[mask] = self.PAD_TOKEN_ID
        
        strings: list[str] = []
        
        label_tensor: torch.Tensor
        for label_tensor in _restored_label:
            label_str: str = self.processor.decode(
                label_tensor, 
                skip_special_tokens=True
            )
            strings.append(label_str)
        
        return strings

from transformers import VisionEncoderDecoderModel, PreTrainedModel, PretrainedConfig
from torch import nn

class TrocrApl(nn.Module):
    
    model_name: str = "microsoft/trocr-base-handwritten"
    
    def __init__(self, num_classes: int):
        """
        Initializes the TrOCR model with a custom number of output classes.

        Args:
            num_classes (int): Number of output classes (e.g., unique characters).
        """
        super(TrocrApl, self).__init__()
        
        
        self.num_classes = num_classes
        
        self.model: nn.Module = VisionEncoderDecoderModel.from_pretrained(
            self.model_name
        )
        self.model_config: PretrainedConfig = self.model.config
        self.model_config.decoder.vocab_size = self.num_classes
        self.model.decoder.resize_token_embeddings(self.num_classes)

    def forward(self, pixel_values, labels=None):
        """
        Forward pass for the model.

        Args:
            pixel_values (Tensor): Image tensors with shape (batch_size, channels, height, width).
            labels (Tensor, optional): Label tensors with shape (batch_size, sequence_length).

        Returns:
            dict: Model outputs including loss (if labels are provided) and logits.
        """
        outputs = self.model(pixel_values=pixel_values, labels=labels)
        return outputs

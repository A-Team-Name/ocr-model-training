import torch
import torch.nn as nn
from .allcnn2d import AllCNN2D
from torch.nn import RNN, LSTM, GRU


class CNNRNNModel(nn.Module):
    def __init__(
        self,
        cnn_encoder: AllCNN2D,
        rnn_type: str = 'lstm',
        rnn_hidden_size: int = 128,
        rnn_num_layers: int = 2,
        rnn_dropout: float = 0.0,
        num_classes: int = 22,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        super(CNNRNNModel, self).__init__()

        self.cnn_encoder = cnn_encoder
        self.rnn_type = rnn_type.lower()
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_num_layers = rnn_num_layers
        self.rnn_dropout = rnn_dropout
        self.num_classes = num_classes
        self.device = device

        # Determine the input size for the RNN
        self.cnn_output_size = self.cnn_encoder.conv_latent_size

        # Define the RNN
        if self.rnn_type == 'lstm':
            self.rnn = LSTM(
                input_size=self.cnn_output_size,
                hidden_size=self.rnn_hidden_size,
                num_layers=self.rnn_num_layers,
                dropout=self.rnn_dropout if self.rnn_num_layers > 1 else 0,
                batch_first=True
            )
        elif self.rnn_type == 'gru':
            self.rnn = GRU(
                input_size=self.cnn_output_size,
                hidden_size=self.rnn_hidden_size,
                num_layers=self.rnn_num_layers,
                dropout=self.rnn_dropout if self.rnn_num_layers > 1 else 0,
                batch_first=True
            )
        else:
            self.rnn = RNN(
                input_size=self.cnn_output_size,
                hidden_size=self.rnn_hidden_size,
                num_layers=self.rnn_num_layers,
                dropout=self.rnn_dropout if self.rnn_num_layers > 1 else 0,
                batch_first=True
            )

        # Define the final fully connected layer
        self.fc = nn.Linear(self.rnn_hidden_size, self.num_classes)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            X (torch.Tensor): Input tensor of shape (batch_size, sequence_length, channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, sequence_length, num_classes).
        """
        batch_size, seq_len, channels, height, width = X.size()

        # Pass each frame in the sequence through the CNN encoder
        X = X.view(batch_size * seq_len, channels, height, width)

        conv_block: torch.nn.Module
        for conv_block in self.cnn_encoder.encoder_conv_blocks:
            X = conv_block(X)

        X = self.cnn_encoder.conv_flatten_layer(X)

        cnn_output = X.view(batch_size, seq_len, -1)  # Reshape back to (batch_size, seq_len, cnn_output_size)
        print(cnn_output.shape)  # batch, seq, 256
        # Pass the CNN output through the RNN
        if self.rnn_type == 'lstm':
            rnn_output, (hidden, cell) = self.rnn(cnn_output)
        else:
            rnn_output, hidden = self.rnn(cnn_output)

        # Pass the RNN output through the final fully connected layer
        # Apply the fully connected layer to each time step's output
        output = self.fc(rnn_output)  # Shape: (batch_size, seq_len, num_classes)

        return output

import os
from torch import dtype as torch_dtype
from torch import float32 as torch_float32
from torch import where as torch_where
from torch import isnan as torch_isnan
from torch import zeros_like as torch_zeros_like
from torch import Tensor, tensor
from torch.cuda import is_available as cuda_is_available
from torch.nn import LeakyReLU
from torch.nn import BatchNorm2d
from torch.nn import Sequential
from torch.nn import AdaptiveAvgPool2d
from torch.nn import Module
from torch.nn import Flatten
from torch.nn import Linear
from torch.nn import Conv2d
from torch.nn import Dropout2d, Dropout
from torch.nn import ModuleList
from torch.nn import Tanhshrink
from torch import stack
from torch import load as torch_load
import torch.functional as F
from torch.nn.functional import softmax
from torchinfo import summary
from typing import Any, Callable, Tuple, Type
from numpy import ceil, prod
from datetime import datetime


class LambdaBlock(Module):
    def __init__(
        self,
        func: Callable
    ):
        super().__init__()

        self.func: Callable = func

    def forward(
        self,
        X: Tensor
    ) -> Tensor:
        return self.func(X)


class AllCNN2D(Module):

    def __init__(
        self,
        conv_features: tuple[int, ...],
        fully_connected_features: tuple[int, ...],
        expected_input_size: tuple[int, ...],
        dtype: torch_dtype = torch_float32,
        conv_dropout: float = 0.0,
        fully_connected_dropout: float = 0.0,
        leaky_gradient: float = 0.05,
        leaky_inplace: bool = True,
        device: str | None = None,
        name_prefix: str = "AllCNN2D",
        checkpoint_path: str | None = None,
        frozen_layer_prefixes: list[str] = [],
        conv_latent_activation_factory: Callable[[], Module] | None = None,
        fc_activation_factory: Callable[[], Module] | None = None,
        verbose=True
    ):
        super(AllCNN2D, self).__init__()

        """All CNN model, generalised and adapted to fit a 2D image context.

        Parameters:
            conv_features               (tuple[int, ...])   : Tuple of convolution feature sizes.
            fully_connected_features    (tuple[int, ...])   : Tuple of fully connected feature sizes.
            expected_input_size         (tuple[int, ...])   : Expected shape of the input (Optional[c], y, x).
            dtype                       (torch_dtype)       : Datatype. Default is `torch_float32`.
            conv_dropout                (float)             : Convolution block dropout. Default is `0.0`.
            fully_connected_dropout     (float)             : Fully connected block dropout. Default is `0.0`.
            leaky_gradient              (float)             : Leaky ReLU gradient. Default is `0.05`.
            leaky_inplace               (bool)              : Default is `True`.
            device                      (str | None)        : Default is `None`.
            name_prefix                 (str)               : Default is "AllCNN2D".
            checkpoint_path             (str | None)        : Default is None.
        """

        if len(conv_features) <= 2:
            raise Exception("Conv encoder needs at least two features")

        self.device: str
        if device is None:
            self.device = "cuda" if cuda_is_available() else "cpu"
        else:
            self.device = device

        if len(expected_input_size) <= 1:
            raise Exception(
                "expected input size should be shape (Optional[c], y, x)"
            )

        if len(expected_input_size) == 2:
            expected_input_size = (1, *expected_input_size)

        if conv_latent_activation_factory is None:
            def conv_latent_activation_factory():
                return LeakyReLU(
                    self.leaky_gradient,
                    self.leaky_inplace
                )

        if fc_activation_factory is None:
            def fc_activation_factory():
                return LeakyReLU(
                    self.leaky_gradient,
                    self.leaky_inplace
                )

        self.name: str = name_prefix
        self.date_created: datetime = datetime.now()
        self.conv_features: tuple[int, ...] = conv_features
        self.fully_connected_features: tuple[int, ...]
        self.fully_connected_features = fully_connected_features
        self.dtype: torch_dtype = dtype
        self.conv_dropout: float = conv_dropout
        self.fully_connected_dropout: float = fully_connected_dropout
        self.leaky_gradient: float = abs(leaky_gradient)
        self.leaky_inplace: bool = leaky_inplace
        self.expected_input_size: tuple[int, int, int]
        self.expected_input_size = expected_input_size[-3:]
        self.checkpoint_path: str = checkpoint_path
        self.encoder_conv_blocks: ModuleList = ModuleList()
        self.fully_connected_blocks: ModuleList = ModuleList()
        self.expected_conv_shapes: list[tuple[int, int, int]]
        self.expected_conv_shapes = [expected_input_size]
        self.verbose: bool = verbose
        self.frozen_layer_prefixes: list[str] = frozen_layer_prefixes

        in_feature: int
        out_feature: int
        for i, in_feature, out_feature in zip(
            range(len(conv_features)-1),
            conv_features[:-1],
            conv_features[1:]
        ):
            input_shape: tuple[int, int, int]
            input_shape = self.expected_conv_shapes[-1]

            output_shape: tuple[int, int, int] = (
                out_feature,
                int(ceil(input_shape[1]/2)),
                int(ceil(input_shape[2]/2))
            )

            self.expected_conv_shapes.append(
                output_shape
            )

            out_activation_factory: Callable[[], Module] | None = None

            if i == len(conv_features)-2:
                out_activation_factory = conv_latent_activation_factory

            conv_block = self._make_conv_block(
                in_features=in_feature,
                out_features=out_feature,
                out_activation_factory=out_activation_factory
            )

            self.encoder_conv_blocks.append(
                conv_block
            )

        self.conv_latent_size: int = prod(
            self.expected_conv_shapes[-1]
        )

        self.conv_flatten_layer: Sequential = Sequential(
            Flatten(start_dim=-3)
        )
        fully_connected_features: tuple[int] = (
            self.conv_latent_size,
            *fully_connected_features
        )
        for in_feature, out_feature in zip(
            fully_connected_features[:-2],
            fully_connected_features[1:-1]
        ):
            self.fully_connected_blocks.append(
                self._make_fully_connected_block(
                    in_features=in_feature,
                    out_features=out_feature,
                    activation_factory=fc_activation_factory
                )
            )
        self.fully_connected_blocks.append(
            self._make_fully_connected_block(
                in_features=fully_connected_features[-2],
                out_features=fully_connected_features[-1],
                include_activation=False
            )
        )

        if checkpoint_path is not None \
                and os.path.exists(checkpoint_path):
            self.load_state(
                checkpoint_path,
                verbose=self.verbose
            )

        self = self.to(
            device=self.device,
            dtype=dtype
        )

        for prefix in frozen_layer_prefixes:
            self.freeze_layers(prefix)

        if verbose:
            print(
                str(
                    summary(
                        self,
                        (1, *expected_input_size),
                        device=device
                    )
                )
            )

    def freeze_layers(
        self,
        prefix: str,
        verbose: bool = False
    ) -> None:
        """
        Freeze all layers in a model whose names start with the given prefix.

        Args:
            prefix (str): The prefix to match layer names.
            verbose (bool): ...
        """
        for name, param in self.named_parameters():
            if name.startswith(prefix):
                param.requires_grad = False
                if verbose:
                    print(f"Froze layer: {name}")

    def unfreeze_layers(
        self,
        prefix: str,
        verbose: bool = False
    ) -> None:
        """
        Freeze all layers in a model whose names start with the given prefix.

        Args:
            prefix (str): The prefix to match layer names.
            verbose (bool): ...
        """
        for name, param in self.named_parameters():
            if name.startswith(prefix):
                param.requires_grad = True
                if verbose:
                    print(f"Unfroze layer: {name}")

    def load_state(
        self,
        checkpoint_path: str,
        verbose: bool = True
    ) -> None:
        """Loads a checkpoint from a filepath.

        Args:
            checkpoint_path (str): File path to the checkpoint.
            verbose (bool, optional): Defaults to True.
        """
        checkpoint: dict[str, Tensor]
        with open(checkpoint_path, "rb") as checkpoint_file:
            try:
                checkpoint = torch_load(
                    checkpoint_file,
                    weights_only=True,
                    map_location='cpu'
                )
            except Exception:
                checkpoint = torch_load(
                    checkpoint_file,
                    weights_only=False
                )

        # Get the model's state_dict
        model_state_dict: dict[str, Tensor] = self.state_dict()

        # Filter checkpoint keys with matching sizes
        filtered_state_dict = {}
        for key, param in checkpoint.items():
            if key not in model_state_dict:
                continue

            if model_state_dict[key].size() == param.size():
                filtered_state_dict[key] = param
                if verbose:
                    print(f"Loaded: {key}")
            else:
                if verbose:
                    print(
                        f"Skipping {key}: ",
                        f"loaded size:{param.size()}",
                        "!= model size: ",
                        f"{model_state_dict[key].size()}"
                    )

        # Load the filtered state_dict
        self.load_state_dict(
            filtered_state_dict,
            strict=False
        )

    def get_name(
        self
    ) -> str:
        """Creates a string representation of the model's hyperparameters.
        Can be used as a filename or as a unique (hashable) identifier.

        Returns:
            str: Model name.
        """
        return f"{self.name}_\
{self.date_created.strftime('%Y-%m-%d_%H-%M-%S')}_\
Conv{'-'.join(str(f) for f in self.conv_features)}_\
ConvDropout{self.conv_dropout}_\
FC{'-'.join(str(f) for f in self.fully_connected_features)}_\
FCDropout{self.fully_connected_dropout}_\
LeakyGrad{self.leaky_gradient}"

    def replace_nan(
        self,
        X: Tensor
    ) -> Tensor:
        return torch_where(
            torch_isnan(X),
            torch_zeros_like(X),
            X
        )

    def _make_conv_block(
        self,
        in_features: int,
        out_features: int,
        out_activation_factory: Callable[[], Module] | None = None
    ) -> Sequential:

        if out_activation_factory is None:
            def out_activation_factory(): return LeakyReLU(
                self.leaky_gradient,
                inplace=self.leaky_inplace
            )

        return Sequential(
            Conv2d(
                in_channels=in_features,
                out_channels=out_features,
                kernel_size=3,
                padding=1
            ),
            Dropout2d(
                self.conv_dropout
            ),
            BatchNorm2d(
                out_features
            ),
            LeakyReLU(
                self.leaky_gradient,
                inplace=self.leaky_inplace
            ),
            Conv2d(
                in_channels=out_features,
                out_channels=out_features,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            Dropout2d(
                self.conv_dropout
            ),
            BatchNorm2d(
                out_features
            ),
            out_activation_factory()
        )

    def _make_fully_connected_block(
        self,
        in_features: int,
        out_features: int,
        activation_factory: Callable[[], Module] | None = None,
        include_activation: bool = True
    ) -> Sequential:
        """Creates a fully connected block as a `Sequential` module.

        Args:
            in_features (int)
            out_features (int)
            include_activation (bool, optional): Defaults to True.

        Returns:
            Sequential
        """
        if not include_activation:
            return Sequential(
                Linear(
                    in_features=in_features,
                    out_features=out_features
                ),
            )

        if activation_factory is None:
            def activation_factory():
                return LeakyReLU(
                    self.leaky_gradient,
                    self.leaky_inplace
                )

        return Sequential(

            Linear(
                in_features=in_features,
                out_features=out_features
            ),
            Dropout(
                self.fully_connected_dropout
            ),
            activation_factory(),
        )

    def forward(
        self,
        X: Tensor
    ) -> Tensor:
        """Forward pass through the model

        Args:
            X (Tensor): shape (excluding batch) should match `self.expected_input_size`

        Returns:
            Tensor: batched y_hat
        """

        X = X.to(
            device=self.device,
            dtype=self.dtype
        )

        conv_block: Module
        for conv_block in self.encoder_conv_blocks:
            X = conv_block(X)

        X = self.conv_flatten_layer(X)

        fc_block: Module
        for fc_block in self.fully_connected_blocks:
            X = fc_block(X)

        return X

    def forward_until(
        self,
        X: Tensor,
        until_block: int
    ) -> Tensor:
        """Forward pass through the model until a block

        Args:
            X (Tensor): shape (excluding batch) should match `self.expected_input_size`
            until_block (int): the block to forward pass up to (inclusive)
        Returns:
            Tensor: batched block output
        """

        if until_block < 0:
            until_block = len(self.encoder_conv_blocks) + \
                len(self.fully_connected_blocks) + until_block

        num_conv_blocks_iter: int = min(
            len(self.encoder_conv_blocks),
            until_block+1
        )

        num_fc_blocks_iter: int = min(
            len(self.fully_connected_blocks),
            until_block + 1 - len(self.encoder_conv_blocks)
        )

        X = X.to(
            device=self.device,
            dtype=self.dtype
        )

        conv_block: Sequential
        for i in range(num_conv_blocks_iter):
            conv_block = self.encoder_conv_blocks[i]
            X = conv_block(X)

        X = self.conv_flatten_layer(X)

        fc_block: Sequential
        for i in range(num_fc_blocks_iter):
            fc_block = self.fully_connected_blocks[i]
            X = fc_block(X)

        layer_weights: Tensor | None
        if num_fc_blocks_iter == len(self.fully_connected_blocks):
            layer_weights = None

        elif num_conv_blocks_iter == len(self.encoder_conv_blocks):
            fc_block: Sequential = self.fully_connected_blocks[
                num_fc_blocks_iter
            ]
            layer_weights = fc_block[0].weight

        else:
            layer_weights = self.encoder_conv_blocks[
                num_conv_blocks_iter
            ][0].weight

        X = X.detach().clone()
        if layer_weights is not None:
            layer_weights = layer_weights.detach().clone()

        return X, layer_weights

    def predict(
        self,
        X: Tensor
    ) -> Tensor:
        return F.softmax(
            self.forward(X),
            dim=-1
        )

    def get_summary(
        self
    ) -> str:
        return str(
            summary(
                self,
                (1, *self.expected_input_size),
                device=self.device
            )
        )


class AllCNN2D_Prod(AllCNN2D):

    def __init__(
        self,
        labels_map: list[str],
        **kwargs
    ):
        self.labels_map: list[str] = labels_map
        super(AllCNN2D_Prod, self).__init__(**kwargs)  # Correctly call the parent class's __init__

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        # Call the parent class's forward method to get logits
        logits: Tensor = super(AllCNN2D_Prod, self).forward(x)  # Use super to call the parent class's forward method
        softmaxed: Tensor = softmax(logits, dim=-1)  # Use dim instead of axis for PyTorch

        softmaxed_ordered_stack: list[Tensor] = []
        softmaxed_char: Tensor
        for batch_i in range(softmaxed.shape[0]):
            softmaxed_char = softmaxed[batch_i, :]
            softmaxed_char_list: list = [
                (i, pred, ord(char))
                for i, pred, char in
                zip(
                    range(softmaxed_char.shape[0]),
                    softmaxed_char.tolist(),
                    self.labels_map
                )
            ]
            softmaxed_char_list = sorted(
                softmaxed_char_list,
                key=lambda e: e[1],  # prob
                reverse=True
            )

            softmaxed_ordered_stack.append(softmaxed_char_list)

        softmax_ordered: Tensor = tensor(
            softmaxed_ordered_stack
        )

        return logits, softmaxed, softmax_ordered

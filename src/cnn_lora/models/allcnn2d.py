import os
from torch.nn import init as torch_init
from torch import zeros as torch_zeros
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
from torch.nn import Parameter
from torch import stack
from torch import load as torch_load
import torch.functional as F
from torch.nn.functional import softmax
from torchinfo import summary
from typing import Any, Callable, Tuple, Type
from numpy import ceil, prod
from datetime import datetime
import math


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


class LoRAConv2DWrapper(Module):
    def __init__(
        self,
        original: Conv2d,
        freeze_original: bool = True,
        device: str | None = None
    ):
        super().__init__()

        self.device = device or ("cuda" if cuda_is_available() else "cpu")
        self.original = original.to(self.device)

        self.lora_adapters: dict[str, dict[str, Module]] = {}
        self.active_scalings: dict[str, float] = {}
        self._all_scalings: dict[str, float] = {}

        for param in self.original.parameters():
            param.requires_grad = not freeze_original

    def add_lora(
        self,
        name: str,
        rank: int = 4,
        alpha: float = 1.0
    ):
        in_channels = self.original.in_channels
        out_channels = self.original.out_channels
        stride = self.original.stride
        padding = self.original.padding
        dilation = self.original.dilation

        A_conv = Conv2d(
            in_channels, rank,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        ).to(self.device)

        B_conv = Conv2d(
            rank, out_channels,
            kernel_size=1,
            stride=stride,
            padding=0,  # ✅ Padding should be zero since kernel_size=1
            bias=False
        )

        torch_init.kaiming_uniform_(A_conv.weight, a=math.sqrt(5))
        torch_init.kaiming_uniform_(B_conv.weight, a=math.sqrt(5))

        self.lora_adapters[name] = {"A": A_conv, "B": B_conv}
        self.active_scalings[name] = alpha / rank
        self._all_scalings[name] = alpha / rank

        self.add_module(f"{name}_conv2d_lora_A", A_conv)
        self.add_module(f"{name}_conv2d_lora_B", B_conv)

    def set_active_lora(self, name: str):
        for key in self.active_scalings:
            self.active_scalings[key] = 0.0
        if name in self._all_scalings:
            self.active_scalings[name] = self._all_scalings[name]

    def set_lora_scaling(self, name: str, scaling: float):
        if name in self.active_scalings:
            self.active_scalings[name] = scaling

    def forward(self, x: Tensor) -> Tensor:
        x = x.to(self.device)
        out = self.original(x)
        for name, adapter in self.lora_adapters.items():
            scaling = self.active_scalings.get(name, 0.0)
            if scaling > 0.0:
                A = adapter["A"]
                B = adapter["B"]
                lora_out = B(A(x))  # mimic LoRA residual path
                out = out + scaling * lora_out
        return out


class LoRALinearWrapper(Module):
    def __init__(
        self,
        original: Linear,
        freeze_original: bool = True,
        device: str | None = None
    ):

        super().__init__()

        if device is None:
            self.device = "cuda" if cuda_is_available() else "cpu"
        else:
            self.device = device

        self.original = original.to(device=device)
        self.lora_adapters: dict[str, dict[str, Parameter]] = {}  # name → {"A": A, "B": B}
        self.active_scalings: dict[str, float] = {}  # name → scaling
        self._all_scalings: dict[str, float] = {}

        for param in self.original.parameters():
            param.requires_grad = not freeze_original

    def add_lora(
        self,
        name: str,
        A: Parameter | None = None,
        B: Parameter | None = None,
        rank: int = 4,
        alpha: float = 1.0
    ):
        if A is None or B is None:
            A = Parameter(torch_zeros((rank, self.original.in_features), device=self.device))
            B = Parameter(torch_zeros((self.original.out_features, rank), device=self.device))
            torch_init.kaiming_uniform_(A, a=math.sqrt(5))
            torch_init.kaiming_uniform_(B, a=math.sqrt(5))
        else:
            A = A.to(self.device)
            B = B.to(self.device)

        self.lora_adapters[name] = {"A": A, "B": B}
        self.active_scalings[name] = alpha / rank
        self._all_scalings[name] = alpha / rank

        # Register with PyTorch so they're tracked
        self.register_parameter(f"{name}_linear_lora_A", A)
        self.register_parameter(f"{name}_linear_lora_B", B)

    def set_active_lora(self, name: str):
        for key in self.active_scalings:
            self.active_scalings[key] = 0.0
        if name in self.lora_adapters:
            self.active_scalings[name] = self._all_scalings[name]

    def set_lora_scaling(self, name: str, scaling: float):
        if name in self.active_scalings:
            self.active_scalings[name] = scaling

    def forward(self, x: Tensor) -> Tensor:
        x = x.to(self.device)
        out = self.original(x)
        for name, adapter in self.lora_adapters.items():
            A = adapter["A"]
            B = adapter["B"]
            scaling = self.active_scalings.get(name, 0.0)
            if scaling != 0.0:
                out = out + scaling * ((x @ A.T) @ B.T)
        return out


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
        verbose: bool = True,
        use_lora: bool = False,
        lora_configs: list[dict] = [],  # {"rank": ..., "name": ..., "alpha": ..., "path": ...}
        default_lora_name: str = "DEFAULT",
        default_lora_rank: int = 1,
        default_lora_alpha: float = 1.0
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
        self.use_lora: bool = use_lora
        self.lora_configs: list[dict] = lora_configs
        self.default_lora_name: str = default_lora_name
        self.default_lora_rank: int = default_lora_rank
        self.default_lora_alpha: float = default_lora_alpha

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
        else:
            print(f"Checkpoint not found {checkpoint_path}")
        self = self.to(
            device=self.device,
            dtype=dtype
        )

        for prefix in frozen_layer_prefixes:
            self.freeze_layers(prefix)

        if verbose:
            print(str(summary(
                self,
                (1, *expected_input_size),
                device=device
            )))

        if self.use_lora:
            for block in self.fully_connected_blocks:
                for module in block:
                    if isinstance(module, LoRALinearWrapper):
                        for lora_cfg in lora_configs:
                            name = lora_cfg["name"]
                            rank = lora_cfg.get("rank", self.default_lora_rank)
                            alpha = lora_cfg.get("alpha", self.default_lora_alpha)
                            module.add_lora(
                                name=name,
                                rank=rank,
                                alpha=alpha
                            )

                            if "path" in lora_cfg:
                                state_dict = torch_load(
                                    lora_cfg["path"],
                                    map_location="cpu"
                                )
                                own_keys = {
                                    k: v
                                    for k, v in state_dict.items()
                                    if name in k
                                }
                                module.load_state_dict(
                                    own_keys,
                                    strict=False
                                )

                        module.set_active_lora(self.default_lora_name)

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
            def out_activation_factory():
                return LeakyReLU(
                    self.leaky_gradient,
                    inplace=self.leaky_inplace
                )

        # Create first conv
        conv1 = Conv2d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=3,
            padding=1
        )
        # Create second conv
        conv2 = Conv2d(
            in_channels=out_features,
            out_channels=out_features,
            kernel_size=3,
            stride=2,
            padding=1
        )

        # Wrap with LoRA if enabled
        if self.use_lora:
            conv1 = LoRAConv2DWrapper(conv1, device=self.device)
            conv2 = LoRAConv2DWrapper(conv2, device=self.device)

            for cfg in self.lora_configs:
                name = cfg["name"]
                rank = cfg.get("rank", self.default_lora_rank)
                alpha = cfg.get("alpha", self.default_lora_alpha)

                conv1.add_lora(name, rank=rank, alpha=alpha)
                conv2.add_lora(name, rank=rank, alpha=alpha)

            conv1.set_active_lora(self.default_lora_name)
            conv2.set_active_lora(self.default_lora_name)

        return Sequential(
            conv1,
            Dropout2d(self.conv_dropout),
            BatchNorm2d(out_features),
            LeakyReLU(self.leaky_gradient, inplace=self.leaky_inplace),
            conv2,
            Dropout2d(self.conv_dropout),
            BatchNorm2d(out_features),
            out_activation_factory()
        )

    def _make_fully_connected_block(
        self,
        in_features: int,
        out_features: int,
        activation_factory: Callable[[], Module] | None = None,
        include_activation: bool = True
    ) -> Sequential:
        if activation_factory is None:
            def activation_factory():
                return LeakyReLU(
                    self.leaky_gradient,
                    self.leaky_inplace
                )

        if self.use_lora:
            linear = Linear(in_features, out_features)
            linear_layer = LoRALinearWrapper(
                original=linear,
                device=self.device
            )

            # Add all LoRA configs
            for lora_cfg in self.lora_configs:
                name = lora_cfg["name"]
                rank = lora_cfg.get("rank", self.default_lora_rank)
                alpha = lora_cfg.get("alpha", self.default_lora_alpha)
                linear_layer.add_lora(name=name, rank=rank, alpha=alpha)

                if "path" in lora_cfg:
                    state_dict = torch_load(lora_cfg["path"], map_location="cpu")
                    own_keys = {
                        k: v for k, v in state_dict.items()
                        if f"{name}_lora_" in k
                    }
                    linear_layer.load_state_dict(own_keys, strict=False)

            linear_layer.set_active_lora(self.default_lora_name)

        else:
            linear_layer = Linear(
                in_features=in_features,
                out_features=out_features
            )

        if not include_activation:
            return Sequential(linear_layer)

        return Sequential(
            linear_layer,
            Dropout(self.fully_connected_dropout),
            activation_factory()
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

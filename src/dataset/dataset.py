from torch.utils.data import Dataset
from torch import Tensor, tensor
from torch import zeros as torch_zeros
from torch import dtype as torch_dtype
from torch import float32 as torch_float32
from numpy import ndarray


class AbstractHandwrittenLineOfCodeDataset(Dataset):

    def __init__(
        self,
        dtype: torch_dtype = torch_float32
    ):
        self.dtype: torch_dtype = dtype

    @staticmethod
    def normalise_tensor(
        input_tensor: Tensor
    ) -> Tensor:
        image_min_pixel: float = input_tensor.min().item()
        image_max_pixel: float = input_tensor.max().item()
        return (input_tensor - image_min_pixel) / (image_max_pixel - image_min_pixel)

    def get_text_image(
        self,
        text: str,
    ) -> Tensor:
        raise NotImplementedError()

    def tokenise_label(
        self,
        text: str
    ) -> Tensor:
        raise NotImplementedError()

    def __len__(self) -> int:
        raise NotImplementedError()

    def __getitem__(
        self,
        index: int
    ) -> Tensor:
        raise NotImplementedError()


class HandwrittenLineOfCodeDataset(AbstractHandwrittenLineOfCodeDataset):

    def __init__(
        self,
        dataset_line_text: list[str],
        unicode_character_filepath_map: dict[str, list[str]],
        eol_char: str,
        dtype: torch_dtype = torch_float32
    ):
        """Initialise the handwritten line of code dataset.

        Args:
            dataset_line_text (list[str]): list of all lines of code (doesn't need to have a EOL symbol here)
            unicode_character_filepath_map (dict[str, list[str]]): dict of a unique identifier for each unicode symbol and a list of all files to the images of those symbol images.
            eol_char str: end of line unique identifier.
            dtype (torch_dtype, optional): _description_. Defaults to torch_float32.
        """

        super(HandwrittenLineOfCodeDataset, self).__init__(
            dtype
        )

        self.dataset_line_text: list[str] = dataset_line_text

        self.unicode_character_filepath_map: dict[str, list[str]]
        self.unicode_character_filepath_map = unicode_character_filepath_map

        _unicode_symbols: list[str] = list(
            unicode_character_filepath_map.keys()
        )
        _unicode_symbols.append(eol_char)
        self.unicode_symbols: list[str] = sorted(_unicode_symbols)

        self.number_of_unicode_symbols: int = len(self.unicode_symbols)

        self.unicode_symbols_index_map: dict[str, int] = {
            symbol: self.unicode_symbols.index(symbol)
            for symbol in self.unicode_symbols
        }

        self.eol_char: str = eol_char
        self.dtype: torch_dtype = dtype

    def get_text_image(
        self,
        text: str,
    ) -> Tensor:
        raise NotImplementedError()

    def encode_unicode_char(
        self,
        unicode: str,
    ) -> Tensor:

        if unicode not in self.unicode_symbols_index_map.keys():
            raise KeyError(
                f"Unicode symbol '{unicode}' not found in dataset"
            )

        unicode_index: int = self.unicode_symbols_index_map[unicode]

        unicode_one_hot: Tensor = torch_zeros(
            self.number_of_unicode_symbols
        )
        unicode_one_hot[unicode_index] = 1

        return unicode_one_hot.to(
            dtype=self.dtype
        )

    def encode_unicode_string(
        self,
        string: str
    ) -> Tensor:
        encoded_tensor = torch_zeros(
            len(string),
            self.number_of_unicode_symbols,
            dtype=self.dtype
        )
        for idx, char in enumerate(string):
            encoded_tensor[
                idx,
                self.unicode_symbols_index_map[char]
            ] = 1
        return encoded_tensor

    def tokenise_label(
        self,
        text: str
    ) -> Tensor:
        return self.encode_unicode_string(text)

    def __len__(self) -> int:
        return len(self.dataset_line_text)

    def __getitem__(
        self,
        index: int
    ) -> tuple[Tensor, Tensor]:

        line_text: str = self.dataset_line_text[index]

        if line_text[-1] != self.eol_char:
            line_text = f"{line_text}{self.eol_char}"

        text_image: Tensor = self.get_text_image(
            line_text
        )

        text_image = text_image.permute(2, 0, 1)  # [w, h, c] -> [c, w, h]

        text_image_tensor = self.normalise_tensor(
            text_image_tensor
        )

        text_label_tensor: Tensor = self.tokenise_label(
            string=line_text,
            dtype=self.dtype
        )

        return text_image_tensor, text_label_tensor

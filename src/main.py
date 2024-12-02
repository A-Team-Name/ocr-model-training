import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers import GenerationConfig
from torch.optim import AdamW
from tqdm import tqdm
import datetime

from models.trocr_apl import TrocrApl
from dataset.dataset import HandwrittenTextDataset



print("Defining Constants")

DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
CSV_SEPERATOR: str = "🫤"
MAX_STRING_LENGTH: int = 128
EARLY_STOPPING: bool = True
BEAM_WIDTH: int = 5
BATCH_SIZE: int = 4
LENGTH_PENALTY: float = 2.0
NO_REPEAT_NGRAM: int = 2
START_EPOCH: int = 13


print("Defining Paths")
file_dirpath: str = os.path.abspath(os.path.dirname(__file__)) # os.path.abspath(".") # (Jupyter)
root_dirpath: str = os.path.join(
    file_dirpath,
    os.pardir
)
dataset_dirpath: str = os.path.join(
    root_dirpath,
    "dataset"
)
apl_dataset_dirpath: str = os.path.join(
    dataset_dirpath,
    "apl_dataset"
)
metadata_csv_filepath: str = os.path.join(
    dataset_dirpath,
    "metadata_apl_fix.csv"
)
checkpoint_dirpath: str = os.path.join(
   root_dirpath,
   "models",
   "trocr-apl" 
) #"microsoft/trocr-base-stage1"

#os.makedirs(checkpoint_dirpath, exist_ok=True)

log_dirpath: str = os.path.join(
    root_dirpath,
    "logs"
)
os.makedirs(log_dirpath, exist_ok=True)


def log(
    string: str,
    filename: str = "log.txt",
    max_length: int = 100000
) -> None:
    
    log_filepath: str = os.path.join(
        log_dirpath,
        filename
    )
    
    formatted_string: str = f"{datetime.datetime.now()}{CSV_SEPERATOR}{string}\n"
    
    with open(log_filepath, "a") as f:
        f.write(formatted_string)





print("Loading Model")

trocr_model: TrocrApl = TrocrApl(
    max_target_length=128,
    model_checkpoint_path=checkpoint_dirpath,
    apl_tokeniser_path=checkpoint_dirpath
)








print("Loading MetaData")

metadata_df: pd.DataFrame = pd.read_csv(
    metadata_csv_filepath,
    delimiter=CSV_SEPERATOR,
    encoding="utf-8",
    engine="python"
)
filenames: list[str] = metadata_df["filename"].to_list()
labels: list[str] = metadata_df["label"].to_list()




print("Splitting dataset")


train_filenames: list[str] = []
val_filenames: list[str] = []
train_labels: list[str] = []
val_labels: list[str] = []

train_filenames, val_filenames, train_labels, val_labels = train_test_split(
    filenames, 
    labels,
    train_size=0.99
)




print("Loading Datasets")


train_dataset: HandwrittenTextDataset = HandwrittenTextDataset(
    dataset_dirpath=apl_dataset_dirpath,
    filenames=train_filenames,
    label_strings=train_labels,
    processor=trocr_model.processor,
    max_target_length=MAX_STRING_LENGTH
)
val_dataset: HandwrittenTextDataset = HandwrittenTextDataset(
    dataset_dirpath=apl_dataset_dirpath,
    filenames=val_filenames,
    label_strings=val_labels,
    processor=trocr_model.processor,
    max_target_length=MAX_STRING_LENGTH
)

train_dataloader: DataLoader = DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_dataloader: DataLoader = DataLoader(
    dataset=val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)
print(f"Dataset Split: Train: {len(train_dataset)} Val: {len(val_dataset)}")

print("Defining Generation Config")

gen_config: GenerationConfig = GenerationConfig(
    max_length=MAX_STRING_LENGTH,
    early_stopping=EARLY_STOPPING,
    num_beams=BEAM_WIDTH,
    length_penalty=LENGTH_PENALTY,
    no_repeat_ngram_size=NO_REPEAT_NGRAM
)


print("Defining Optimiser")


optimiser: AdamW = AdamW(
    params=trocr_model.model.parameters(),
    lr=5e-5,
    weight_decay=0.02
)




print("Training Loop")



epoch: int
for epoch in range(10000):
    epoch = START_EPOCH + epoch
    train_loss: float = 0.0
    train_cer: float = 0.0
    
    data: tuple[torch.Tensor, torch.Tensor]
    
    for i, data in tqdm(
        iterable=enumerate(train_dataloader),
        desc="Training model...",
        total=len(train_dataset)//BATCH_SIZE
    ):
        image: torch.Tensor
        encoded_label: torch.Tensor
        image, encoded_label = data
        
        image = image.to(DEVICE)
        encoded_label = encoded_label.to(DEVICE)
        
        trocr_model.model.to(device=DEVICE)
        trocr_model.model = trocr_model.model.train()
        
        trocr_output: Seq2SeqLMOutput = trocr_model.forward(
            pixels=image,
            encoded_labels=encoded_label
        )
        
        predicted_encoded_string: torch.Tensor = trocr_model.model.generate(
            image,
            generation_config=gen_config
        )
        cer: float = trocr_model.compute_character_error_rate(
            y_hat_ids=predicted_encoded_string, 
            y_ids=encoded_label
        )
        predicted_string: str = trocr_model.decode_model_output(
            predicted_encoded_string
        )
        y_string: str = trocr_model.decode_model_output(
            encoded_label
        )
        
        if i % 300 == 0:
            #print(f"Epoch{epoch}: Training: y:{y_string} y_hat{predicted_string}")
            log(f"{epoch}{CSV_SEPERATOR}{y_string}{CSV_SEPERATOR}{predicted_string}", "predictions.txt")
            
        loss: torch.Tensor = trocr_output.loss
        loss.backward()
        optimiser.step()
        optimiser.zero_grad()

        train_loss += loss.item()
        train_cer += cer
        
    #print(f"Epoch{epoch}: Training: Loss:", train_loss/len(train_dataloader))
    log(f"{epoch}{CSV_SEPERATOR}{train_loss/len(train_dataloader)}", "train_loss.txt")
    log(f"{epoch}{CSV_SEPERATOR}{ train_cer / len(train_dataloader)}", "train_cer.txt")
    
    trocr_model.model = trocr_model.model.eval()
    
    val_loss: float = 0.0
    valid_cer: float = 0.0
    with torch.no_grad():
        
        data: tuple[torch.Tensor, torch.Tensor]       
        for i, data in tqdm(
            enumerate(val_dataloader),
            "Validating",
            len(val_dataloader)
        ):  
            
            image: torch.Tensor
            encoded_label: torch.Tensor 
            image, encoded_label = data
            
            image = image.to(device=DEVICE)
            encoded_label = encoded_label.to(device=DEVICE)
            
            trocr_output: Seq2SeqLMOutput = trocr_model.forward(
                pixels=image,
                encoded_labels=encoded_label
            )
            
            loss: torch.Tensor = trocr_output.loss
            
            encoded_string_prediction: torch.Tensor = trocr_model.model.generate(
                image,
                generation_config=gen_config
            )
            
        
            predicted_output: list[str] = trocr_model.decode_model_output(
                encoded_string_prediction
            )
            correct_output: list[str] = trocr_model.decode_model_output(
                encoded_label
            )
            
            #plt.imshow(X.detach().cpu()[0, :, :, :].permute((1, 2, 0)))
            #plt.show()S
            if i % 300:
                #print(f"Epoch{epoch}: Validation: y: {correct_output} y_hat: {predicted_output}")
                log(f"{epoch}{CSV_SEPERATOR}{correct_output}{CSV_SEPERATOR}{predicted_output}", "val_predictions.txt")
        
            cer: float = trocr_model.compute_character_error_rate(
                y_hat_ids=encoded_string_prediction, 
                y_ids=encoded_label
            )
            
            val_loss += loss.item()
            valid_cer += cer 

    #print("Validation CER:", valid_cer / len(val_dataloader))
    log(f"{epoch}{CSV_SEPERATOR}{ valid_cer / len(val_dataloader)}", "val_cer.txt")
    log(f"Epoch {epoch}: Validation Loss: {val_loss / len(val_dataloader)}")    
    log(f"{epoch}{CSV_SEPERATOR}{val_loss / len(val_dataloader)}", "val_loss.txt")
    trocr_model.model.save_pretrained(checkpoint_dirpath)
    trocr_model.processor.save_pretrained(checkpoint_dirpath)
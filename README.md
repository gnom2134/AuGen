# AuGen
Generative augmentation of medical data.

## Usage in Google Colab

### Model train pipeline

```python
!git clone https://github.com/gnom2134/AuGen.git
!pip install torchxrayvision pydicom pytorch-lightning wandb 1> /dev/null


from google.colab import drive
drive.mount('/content/drive')


!unzip -qq "SIIM.zip path" -d ./SIIM_data


from AuGen.src.train import train


train(
    "siim", 
    "./SIIM_data/SIIM_TRAIN_TEST", 
    "./SIIM_data/SIIM_TRAIN_TEST/train-rle.csv",
    model_name="CVAE",
    model_params={"n_classes": 2, "embedding_dim": 32, "latent_dim": 32},
    trainer_params={"max_epochs": 1}
)
```

### You can also just extract models

```python
from AuGen.src.models import CGANModel, CVAEModel
```

[Link to colab notebook](https://colab.research.google.com/drive/1Yay3o_s786abFHwIOFRrAfTTIVj_7nGL?usp=sharing)

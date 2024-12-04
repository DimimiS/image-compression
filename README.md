# Image Compression

This project implements an image compression model using a deep neural network with rate-distorion loss. The model architecture is an autoencoder with 4 convolutional layers on encoder and decoder. 
The code is using the pytorch library along with several classes and methods from the compressai library.

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/DimimiS/image-compression
   cd image-compression
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Download and prepare your dataset in the `data/` directory.

## Training

To train the model, run:
```bash
python train.py ["lamda-decimal-value (001)"]
```
As mentioned you can include a string that represents the value of the lambda parameter after the decimal point. For instance, for lambda = 0.001 then the input should be 001.

## Evaluation

To evaluate the model on the test dataset, run:
```bash
python evaluate.py ["dataset-path (images/test/)"]
```
Where images/test is the path to your test images.

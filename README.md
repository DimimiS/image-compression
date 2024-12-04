# Image Compression

This project implements an image compression model using a deep neural network with perceptual loss and MS-SSIM. The model architecture includes residual blocks to improve performance.

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

## Evaluation

To evaluate the model on the test dataset, run:
```bash
python evaluate.py ["dataset-path images/test/"]
```
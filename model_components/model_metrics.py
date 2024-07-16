import tensorflow as tf
import matplotlib.pyplot as plt

# Define the PSNR and MS-SSIM metrics
def psnr(y_true, y_pred):
    return tf.image.psnr(y_true, y_pred, max_val=1.0)

def ms_ssim(y_true, y_pred):
    return tf.image.ssim_multiscale(y_true, y_pred, max_val=1.0)

# Custom callback to log metrics
class MetricsLogger(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.psnr = []
        self.ms_ssim = []

    def on_epoch_end(self, epoch, logs=None):
        val_psnr = logs.get('val_psnr')
        val_ms_ssim = logs.get('val_ms_ssim')

        self.psnr.append(val_psnr)
        self.ms_ssim.append(val_ms_ssim)

        print(f'Epoch {epoch+1} - PSNR: {val_psnr}, MS-SSIM: {val_ms_ssim}')

# Plot the PSNR and MS-SSIM metrics
def plot_metrics(metrics_logger):
    # Plot the PSNR and MS-SSIM metrics
    epochs = range(1, len(metrics_logger.psnr) + 1)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, metrics_logger.psnr, label='PSNR')
    plt.title('PSNR Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('PSNR')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, metrics_logger.ms_ssim, label='MS-SSIM')
    plt.title('MS-SSIM Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('MS-SSIM')
    plt.legend()

    plt.tight_layout()
    plt.show()
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import glob

def read_png(filename):
  """Loads a PNG image file."""
  string = tf.io.read_file(filename)
  return tf.image.decode_image(string, channels=3)


def write_png(filename, image):
  """Saves an image to a PNG file."""
  string = tf.image.encode_png(image)
  tf.io.write_file(filename, string)


def check_image_size(image, patchsize):
  shape = tf.shape(image)
  return shape[0] >= patchsize and shape[1] >= patchsize and shape[-1] == 3


def crop_image(image, patchsize):
  image = tf.image.random_crop(image, (patchsize, patchsize, 3))
  return tf.cast(image, tf.keras.mixed_precision.global_policy().compute_dtype)


def get_dataset(name, split, args):
  """Creates input data pipeline from a TF Datasets dataset."""
  with tf.device("/cpu:0"):
    dataset = tfds.load(name, split=split, shuffle_files=True)
    if split == "train":
      dataset = dataset.repeat()
    dataset = dataset.filter(
        lambda x: check_image_size(x["image"], args.patchsize))
    dataset = dataset.map(
        lambda x: crop_image(x["image"], args.patchsize))
    dataset = dataset.batch(args.batchsize, drop_remainder=True)
  return dataset


def get_custom_dataset(split, args):
  """Creates input data pipeline from custom PNG images."""
  with tf.device("/cpu:0"):
    files = glob.glob(args.train_glob)
    if not files:
      raise RuntimeError(f"No training images found with glob "
                         f"'{args.train_glob}'.")
    dataset = tf.data.Dataset.from_tensor_slices(files)
    dataset = dataset.shuffle(len(files), reshuffle_each_iteration=True)
    if split == "train":
      dataset = dataset.repeat()
    dataset = dataset.map(
        lambda x: crop_image(read_png(x), args.patchsize),
        num_parallel_calls=args.preprocess_threads)
    dataset = dataset.batch(args.batchsize, drop_remainder=True)
  return dataset

def calculate_metrics(original_image, decompressed_image, original_image_path, decompressed_image_path):
    """Calculates PSNR, SSIM, and BPP metrics."""
    # Ensure images are in uint8 format
    original_image = tf.image.convert_image_dtype(original_image, dtype=tf.uint8).numpy()
    decompressed_image = tf.image.convert_image_dtype(decompressed_image, dtype=tf.uint8).numpy()

    psnr_value = psnr(original_image, decompressed_image)
    ssim_value = ssim(original_image, decompressed_image, win_size=3, channel_axis=-1)  # Set win_size and channel_axis

    # Calculate BPP (assuming the decompressed image is stored in a file)
    original_size = tf.io.gfile.stat(original_image_path).length
    decompressed_size = tf.io.gfile.stat(decompressed_image_path).length
    bpp_value = (decompressed_size * 8) / original_size

    return psnr_value, ssim_value, bpp_value
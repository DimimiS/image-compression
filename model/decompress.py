import tensorflow as tf
import tensorflow_compression as tfc
from util.basic_util import write_png

def decompress(args):
  """Decompresses an image."""
  # Load the model and determine the dtypes of tensors required to decompress.
  model = tf.keras.models.load_model(args.model_path)
  dtypes = [t.dtype for t in model.decompress.input_signature]

  # Read the shape information and compressed string from the binary file,
  # and decompress the image using the model.
  with open(args.input_file, "rb") as f:
    packed = tfc.PackedTensors(f.read())
  tensors = packed.unpack(dtypes)
  x_hat = model.decompress(*tensors)

  # Write reconstructed image out as a PNG file.
  write_png(args.output_file, x_hat)

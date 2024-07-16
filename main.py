import tensorflow as tf
# i,port the necessarey functions, objects and classes from the model_components module
from model_components.model import ImageCompressionModel
from model_components.dataset import train, valid, train_generator, validation_generator
from model_components.model_metrics import MetricsLogger, psnr, ms_ssim, plot_metrics
# # Enable mixed precision
# from tensorflow.keras.mixed_precision import set_global_policy
# set_global_policy('mixed_float16')


train_generator.batch_size = 2
validation_generator.batch_size = 2

# Custom Loss Function: This function calculates the mean absolute error between the true and predicted images
def custom_loss(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_true - y_pred))

# Compile the Model with Adam optimizer and custom loss
input_shape = (320, 320, 3)
model = ImageCompressionModel(input_shape)

model.compile(optimizer='adam', loss='mse', metrics=[psnr, ms_ssim])

# Create an instance of the MetricsLogger callback
metrics_logger = MetricsLogger()

# Train the Model
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    epochs=1,
    callbacks=[metrics_logger]
)

# Evaluate the Model on the validation set
loss = model.evaluate(validation_generator)
print(f"Validation Loss: {loss}")

# Save model
model.save('cnn-gdn.keras')

# Plot the metrics after training
plot_metrics(metrics_logger)
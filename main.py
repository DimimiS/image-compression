import tensorflow as tf
from model_components.model import ImageCompressionModel
from model_components.dataset import train_generator, validation_generator
from model_components.model_metrics import MetricsLogger, psnr, ms_ssim, plot_metrics, mse
from model_components.custom_loss import RateDistortionLoss

train_generator.batch_size = 16
validation_generator.batch_size = 16

rd_loss = RateDistortionLoss()

# Compile the Model with Adam optimizer and custom loss
input_shape = (256, 256, 3)
model = ImageCompressionModel(input_shape)

# optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, clipnorm=1.0)
model.compile(optimizer='adam', loss=rd_loss, metrics=[psnr, ms_ssim, mse])

# Create an instance of the MetricsLogger callback
metrics_logger = MetricsLogger()

# Train the Model
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    epochs=150,
    callbacks=[metrics_logger]
)

# Evaluate the Model on the validation set
loss = model.evaluate(validation_generator)
print(f"Validation Loss: {loss}")

# Save model
model.save('cnn-gdn.keras')

# Plot the metrics after training
plot_metrics(metrics_logger)
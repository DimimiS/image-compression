import tensorflow as tf
# i,port the necessarey functions, objects and classes from the model_components module
from model_components.model import ImageCompressionModel
from model_components.dataset import train_generator, validation_generator

# Custom Loss Function: This function calculates the mean absolute error between the true and predicted images
def custom_loss(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_true - y_pred))

# Compile the Model with Adam optimizer and custom loss
input_shape = (256, 256, 3)
model = ImageCompressionModel(input_shape)
model.compile(optimizer='adam', loss=custom_loss)

# Train the Model
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    epochs=10
)
# Reshape the output of the RNN block to match the expected shape of the synthesis block
x = tf.reshape(x, (-1, 32, 32, 128))

# Evaluate the Model on the validation set
loss = model.evaluate(validation_generator)
print(f"Validation Loss: {loss}")
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, datasets



def prepare_features_and_labels(x, y):
  x = tf.cast(x, tf.float32) / 255.0
  y = tf.cast(y, tf.int64)

  return x, y


(x, y), (x_val, y_val) = datasets.fashion_mnist.load_data()
y = tf.one_hot(y, depth=10)
y_val = tf.one_hot(y_val, depth=10)
ds = tf.data.Dataset.from_tensor_slices((x, y))
ds = ds.map(prepare_features_and_labels)
train_dataset = ds.shuffle(60000).batch(100)

ds_val = tf.data.Dataset.from_tensor_slices((x_val, y_val))
ds_val = ds_val.map(prepare_features_and_labels)
val_dataset = ds_val.shuffle(10000).batch(100)



model = keras.Sequential([
    layers.Reshape(target_shape=(28 * 28,), input_shape=(28, 28)),
    layers.Dense(200, activation='relu'),
    layers.Dense(200, activation='relu'),
    layers.Dense(200, activation='relu'),
    layers.Dense(10)])
model.compile(optimizer=optimizers.Adam(0.001),
              loss=tf.losses.CategoricalCrossentropy(from_logits=True),
               metrics=['accuracy'])

model.fit(train_dataset.repeat(), epochs=30, steps_per_epoch=500,
          validation_data=val_dataset.repeat(),
          validation_steps=2)



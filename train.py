import tensorflow as tf
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--units",
    type=int,
    default=512,
    help="number of first dense layer hidden units"
)
args = parser.parse_args()
units = int(args.units)
print(f"\n---first dense layer hidden unit: {units}---\n")
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(units, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=1)

model.evaluate(x_test,  y_test, verbose=2)

model.save('/models/saved_model')

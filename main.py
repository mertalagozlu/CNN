import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt


class ImageProcessing:

    def __init__(self, _path):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path=_path)
        self._x_train = x_train
        self._y_train = y_train
        self._x_test = x_test
        self._y_test = y_test


    def __str__():
        return "Using mnist dataset create CNN with following characteristics: \n \
        TODO: Conv3x3 & ReLU & Max Pooling with Resolution 28x28, Channels 8, Layers 3 \n \
        TODO: Conv3x3 & ReLU & Max Pooling with Resolution 14x14, Channels 16, Layers 3 \n \
        TODO: Conv3x3 & ReLU Pooling with Resolution 7x7, Channels 32, Layers 2 \n \
        TODO: Flatten with Resolution 7x7, Channels 1568, Layers 1 \n \
        TODO: Dense & ReLU & Dopout(0,2) 1x1568, Channels 128, Layers 3 \n \
        TODO: Dense & Softmax 1x128, Channels 10, Layers 2"

    def plot_sample(self):
        print(np.shape(self._x_train[2]))

        plt.title(self._y_train[2])
        plt.imshow(self._x_train[2], cmap='gray')
        plt.show()

    def preprocess(self):
        """
         Preprocess the data
         """
        self._x_train = self._x_train.astype('float32') / 255.0
        self._x_test = self._x_test.astype('float32') / 255.0

        # Preprocess the target data
        self._y_train = tf.keras.utils.to_categorical(self._y_train, num_classes=10)
        self._y_test = tf.keras.utils.to_categorical(self._y_test, num_classes=10)

        return "Success!"

    def CNN_model(self):
        """
        Build the CNN
        """
        # Input layer
        input_layer = tf.keras.layers.Input(shape=(28, 28, 1))
        # Convolutional layer 1
        conv_layer_1 = tf.keras.layers.Conv2D(filters=8, kernel_size=(3, 3), strides=(1, 1), padding='valid', \
                                              activation='relu')(input_layer)
        # Max pooling layer 1
        max_pool_layer_1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(conv_layer_1)
        # Convolutional layer 2
        conv_layer_2 = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='valid', \
                                              activation='relu')(max_pool_layer_1)
        # Max pooling layer 2
        max_pool_layer_2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(conv_layer_2)
        # Convolutional layer 3
        conv_layer_3 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', \
                                              activation='relu')(max_pool_layer_2)
        # Max pooling layer 3
        max_pool_layer_3 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(conv_layer_3)
        # Flatten layer
        flatten_layer = tf.keras.layers.Flatten()(max_pool_layer_3)
        # Dense layer 1
        dense_layer_1 = tf.keras.layers.Dense(units=128, activation='relu')(flatten_layer)
        # Dropout layer 1
        dropout_layer_1 = tf.keras.layers.Dropout(rate=0.2)(dense_layer_1)
        # Dense layer 2
        dense_layer_2 = tf.keras.layers.Dense(units=128, activation='relu')(dropout_layer_1)
        # Dropout layer 2
        dropout_layer_2 = tf.keras.layers.Dropout(rate=0.2)(dense_layer_2)
        # Output layer
        output_layer = tf.keras.layers.Dense(units=10, activation='softmax')(dropout_layer_2)
        # Model
        model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

        # Compile the model
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        # Train the model
        history = model.fit(self._x_train, self._y_train, epochs=10, batch_size=64, verbose=0,
                            callbacks=[tf.keras.callbacks.History()])

        # Extract the training loss from the history object
        loss = history.history['loss']

        # Plot the training loss
        plt.plot(loss)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.show()

        # Evaluate the model
        model.evaluate(self._x_test, self._y_test)
        return "Success!"


if __name__ == "__main__":
    ImageProcessing("mnist.npz").plot_sample()
    # ImageProcessing("mnist.npz").preprocess()
    # ImageProcessing("mnist.npz").CNN_model()

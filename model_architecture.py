import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, optimizers, callbacks

from helper import process_data

################## MLP ##################
# Testing output_layer <= num_nodes <= size_input_layer
def test_first_layer(data_arr):
    model = models.Sequential([
        layers.Flatten(input_shape=(28, 28)),
        layers.Dense(28, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    model.summary()

    opt = optimizers.Adam(learning_rate=0.001, name='Adam')

    model.compile(optimizer=opt,
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=['accuracy'])

    c1 = callbacks.EarlyStopping(
        monitor='val_loss', min_delta=0, patience=5, verbose=0,
        mode='auto', baseline=None, restore_best_weights=True
    )

    history = model.fit(data_arr[0], data_arr[2], epochs=100, batch_size=128, callbacks=[c1, c2],
                    validation_data=(data_arr[1], data_arr[2]))

    # Plot Training Accuracy vs Iternation
    plt.subplot(211)
    plt.title('Classification Accuracy')
    plt.plot(history.history['accuracy'], color='blue', label='train')
    plt.plot(history.history['val_accuracy'], color='orange', label='validation')
    plt.legend(loc="lower right")
    plt.savefig('classification_accuracy.png')
    plt.show()
    plt.close()

    # Plot Traing Loss vs Iteration
    plt.subplot(211)
    plt.title('Cross Entropy Loss')
    plt.plot(history.history['loss'], color='blue', label='train')
    plt.plot(history.history['val_loss'], color='orange', label='validation')
    plt.legend(loc="upper right")
    plt.savefig('cross_entropy_loss.png')
    plt.show()
    plt.close()



data_dict = process_data()
test_first_layer(data_dict['split_b'])
    

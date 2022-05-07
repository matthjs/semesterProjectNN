import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras # keras is the goto when building NN models
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist # dataset of images of 0-9 (grayscale 28x28 pixels)

def MLP1():
    # sequential API (Very convenient, not very flexible) (one input and one output)
    model = keras.Sequential(
        [
            keras.Input(shape=(28*28)),                                     # input layer = inputs
            layers.Dense(512, activation='relu', name='hidden_layer_1'),    # 512 units in first layer ("Dense" for fully connected layer)
            layers.Dense(256, activation='relu', name='hidden_layer_2'),    # hidden layer 256 units
            layers.Dense(10, name='output_layer')                           # output layer N(u)=y=W^kx^{k-1} 10 nodes corresponding to number 0-9
        ]
    )
    return model

def MLP2():
    # Functional API (A bit more flexible)
    inputs = keras.Input(shape = (784))
    x = layers.Dense(512, activation='relu', name='hidden_layer_1')(inputs)    # pass inputs to first layer
    x = layers.Dense(256, activation='relu', name='hidden_layer_2')(x)         # pass to next layer
    outputs = layers.Dense(10, activation='softmax', name='output_layer')(x)   # pass to output layer

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

if __name__=='__main__':
    # loading data set
    (u_train, y_train), (u_test, y_test) = mnist.load_data()    # loads training and testing data (u_i, y_i)_{i=1,...,N}
                                                                # training set: 60000 28x28 images 
                                                                # -> flatten them to vector such that u = x^0 (0 = input layer)
    u_train = u_train.reshape(-1, 28*28).astype("float64") / 255.0
    u_test = u_test.reshape(-1, 28*28).astype("float64") / 255.0

    model = MLP2()

    # to add layers one by one do model = keras.Sequential() then model.add(), ....

    print(model.summary())

    # configure training part of network
    model.compile(
        loss = keras.losses.SparseCategoricalCrossentropy(from_logits=False),       # loss function (send in from softmax first)
        optimizer = keras.optimizers.Adam(learning_rate=0.001),                     # learning rate
        metrics = ['accuracy'],                                                     # keep track of running accuracy
    )

    # specify concrete training
    model.fit(u_train, y_train, batch_size=32, epochs=5, verbose=2)     # give training data to model
    model.evaluate(u_test, y_test, batch_size=32, verbose=2)            # give testing data to model

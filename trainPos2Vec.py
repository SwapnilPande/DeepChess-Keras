from keras.layers import Dense, Input
from keras.models import Model
from keras import backend as K

import numpy as np

import os

# Number of neurons per layer of the autoencoder
autoencoderLayers = [773, 600, 400, 200, 100]

# Files containing training data
whiteBoardsFile = "./data/white.npy"
blackBoardsFile = "./data/black.npy"

weightsPath = "./data/"

numWhites = 1000000
numBlacks = 1000000

# Retrieve input data from files
print("Loading data from files:")
print("\t" + whiteBoardsFile)
print("\t" + blackBoardsFile)

x = np.zeros((2000000, 773))
x[:1000000] = np.load(whiteBoardsFile)
x[1000000:] = np.load(blackBoardsFile)

# Shuffle the order of x
np.random.shuffle(x)

print("Successfully loaded data")
print()

# List to store weights and biases from autoencoder
weights = []


print("------------ Beginning training ------------")
# Iterate over the autoencoder layers to train one at a time
for i in range(len(autoencoderLayers) - 1):
    print("Training layer with {numNeurons} neurons".format(
        numNeurons = autoencoderLayers[i+1]
    ))

    # Construct autoencoder model
    # Input has the shape of the previous layer
    inputLayer = Input(shape=(autoencoderLayers[i],))
    encodeLayer = Dense(autoencoderLayers[i+1], activation = 'relu')(inputLayer)
    # Ouptut layer has same shape as input layer
    outputLayer = Dense(autoencoderLayers[i], activation = 'relu')(encodeLayer)

    model = Model(inputs=inputLayer, outputs=outputLayer)
    model.compile(optimizer='adam', loss='mse', metrics=['mse'])
    model.fit(x, x, epochs=5, batch_size = 256, shuffle = True)

    # Extract weights
    weights.append(model.layers[1].get_weights())
    # savePath = os.path.join(weightsPath, str(autoencoderLayers[i+1]) + "-weights.npy")
    # np.save(savePath, model.layers[1].get_weights())


    # Keras backend function to get output of hidden layer
    getHiddenOuptut = K.function([model.input],[model.layers[1].output])

    #xNew = np.zeros((2000000, autoencoderLayers[i+1]))
    # Extract output of hidden layer
    # for i in range(int(2000000/200)):
    #     xNew[200*i:200*i+200] = getHiddenOuptut([x[200*i:200*i+200]])[0]
    # x = xNew


    x = getHiddenOuptut([x])[0]
    print(x.shape)

# Build final autoencoder model
inputLayer = Input(shape=(autoencoderLayers[0],))
ae1 = Dense(autoencoderLayers[1], activation='relu',trainable=False)(inputLayer)
ae2 = Dense(autoencoderLayers[2], activation='relu',trainable=False)(ae1)
ae3 = Dense(autoencoderLayers[3], activation='relu',trainable=False)(ae2)
outputLayer = Dense(autoencoderLayers[4], activation='relu',trainable=False)(ae3)
ae = Model(inputLayer, outputLayer)

# Iterate over autoencoder model - skip over input layer
for layer, weight in zip(ae.layers[1:], weights):
    # Set weights for each layer
    layer.set_weights(weight)
    layer.trainable = False

# Save final model
savePath = os.path.join(weightsPath, "Pos2VecModel.h5")
ae.save(savePath)















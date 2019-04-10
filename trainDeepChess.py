from keras.layers import Dense, Input
from keras.models import Model
from keras import backend as K

import numpy as np

import os

# Number of neurons per layer of deepchess
deepchessLayers = [400, 200, 100, 2]

# Path to Pos2Vec Model
pos2VecPath = "./data/Pos2Vec.h5"

numWhites = 1000000
numBlacks = 1000000

# Retrieve input data from files
print("Loading data from files:")
print("\t" + whiteBoardsFile)
print("\t" + blackBoardsFile)

# Load black and white boards from file
whites = np.load(whiteBoardsFile)
blacks = np.load(blackBoardsFile)

print("Successfully loaded data")
print()

# List to store weights and biases from autoencoder
weights = []


print("------------ Beginning training ------------")
# Inputs to Pos2Vec networks
inputLeft = Input(shape=(773,))
inputRight = Input(shape=(773,))
pos2VecLeft =

ae1 = Dense(autoencoderLayers[1], activation='relu',trainable=False)(inputLayer)
ae2 = Dense(autoencoderLayers[2], activation='relu',trainable=False)(ae1)
ae3 = Dense(autoencoderLayers[3], activation='relu',trainable=False)(ae2)
outputLayer = Dense(autoencoderLayers[4], activation='relu',trainable=False)(ae3)
ae = Model(inputLayer, outputLayer)

# Iterate over autoencoder model - skip over input layer
for layer, weight in zip(ae.layers[1:], weights):
    # Set weights for each layer
    layer.set_weights(weight)

# Save final model
savePath = os.path.join(weightsPath, "AutoencoderWeights.h5")
ae.save(savePath)















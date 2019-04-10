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

x = np.zeros((2000, 773))
# x[:1000000] = np.load(whiteBoardsFile)
# x[1000000:] = np.load(blackBoardsFile)

# Shuffle the order of x
np.random.shuffle(x)

print("Successfully loaded data")
print()



print("------------ Beginning training ------------")
# Iterate over the autoencoder layers to train one at a time
#for i in range(len(autoencoderLayers) - 1):
print("Training layer with {numNeurons} neurons".format(
    numNeurons = autoencoderLayers[1]
))


# ---------------------- TRAIN LAYER 1 ----------------------
# Construct autoencoder model
# Input has the shape of the previous layer
inputLayer = Input(shape=(autoencoderLayers[0],))
encodeLayer1 = Dense(autoencoderLayers[1], activation = 'relu')(inputLayer)
# Ouptut layer has same shape as input layer
outputLayer = Dense(autoencoderLayers[0], activation = 'relu')(encodeLayer1)

model = Model(inputs=inputLayer, outputs=outputLayer)
model.summary()

model.compile(optimizer='adam', loss='mse', metrics=['mse'])
model.fit(x, x, epochs=1, batch_size = 256, shuffle = True)
model.layers[1].trainable = False

# ---------------------- TRAIN LAYER 2 ----------------------
encodeLayer2 = Dense(autoencoderLayers[2], activation = 'relu')(encodeLayer1)
# Ouptut layer has same shape as input layer
outputLayer = Dense(autoencoderLayers[1], activation = 'relu')(encodeLayer2)

model = Model(inputs=inputLayer, outputs=outputLayer)
model.summary()

model.compile(optimizer='adam', loss='mse', metrics=['mse'])
model.fit(x, x, epochs=1, batch_size = 256, shuffle = True)
model.layers[2].trainable = False
# ---------------------- TRAIN LAYER 3 ----------------------
encodeLayer3 = Dense(autoencoderLayers[3], activation = 'relu')(encodeLayer2)
# Ouptut layer has same shape as input layer
outputLayer = Dense(autoencoderLayers[2], activation = 'relu')(encodeLayer3)

model = Model(inputs=inputLayer, outputs=outputLayer)
model.compile(optimizer='adam', loss='mse', metrics=['mse'])
model.fit(x, x, epochs=1, batch_size = 256, shuffle = True)
model.layers[3].trainable = False

# ---------------------- TRAIN LAYER 4 ----------------------
encodeLayer4 = Dense(autoencoderLayers[4], activation = 'relu')(encodeLayer3)
# Ouptut layer has same shape as input layer
outputLayer = Dense(autoencoderLayers[3], activation = 'relu')(encodeLayer4)

model = Model(inputs=inputLayer, outputs=outputLayer)
model.compile(optimizer='adam', loss='mse', metrics=['mse'])
model.fit(x, x, epochs=1, batch_size = 256, shuffle = True)
model.layers[4].trainable = False


# Extract weights
# savePath = os.path.join(weightsPath, str(autoencoderLayers[i+1]) + "-weights.npy")
# np.save(savePath, model.layers[1].get_weights())

# # Keras backend function to get output of hidden layer
# getHiddenOuptut = K.function([model.input],[model.layers[1].output])

# xNew = np.zeros((2000000, autoencoderLayers[i+1]))
# # Extract output of hidden layer
# for i in range(int(2000000/200)):
#     xNew[200*i:200*i+200] = getHiddenOuptut([x[200*i:200*i+200]])[0]
# x = xNew
#     print(x.shape)









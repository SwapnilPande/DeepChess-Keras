from keras.layers import Dense, Input, Concatenate
from keras.models import Model, load_model
from keras.utils import Sequence
from keras import backend as K

import numpy as np

import os

os.environ["CUDA_VISIBLE_DEVICES"]="7"
# Number of neurons per layer of deepchess
deepchessLayers = [400, 200, 100, 2]

# Path to Pos2Vec Model
pos2VecPath = "./data/Pos2VecModel.h5"


class DeepchessDataGenerator(Sequence):
    def __init__(self, batchSize):
        self.batchSize = batchSize

        self.whiteBoardsFile = "./data/whites.npy"
        self.blackBoardsFile = "./data/black.npy"

        # Size of dataset
        self.numWhites = 1000000
        self.numBlacks = 1000000

        # Retrieve input data from files
        print("DATA GENERATOR: Loading data from files:")
        print("\t" + self.whiteBoardsFile)
        print("\t" + self.blackBoardsFile)

        # Load black and white boards from file
        self.whites = np.load(whiteBoardsFile)
        self.blacks = np.load(blackBoardsFile)

        # Randomize order of each dataset
        np.random.shuffle(self.whites)
        np.random.shuffle(self.blacks)

        print("DATA GENERATOR: Successfully loaded data")
        print()

    def __len__(self):
        return int(np.ceil(self.numWhites / float(self.batchSize)))

    def __getitem__(self, index):
        startIndex = index*self.batchSize

        try: #Full size batch
            whiteBatch = self.whites[startIndex : startIndex + self.batchSize]
            blackBatch = self.blacks[startIndex : startIndex + self.batchSize]
        except IndexError: #Retrieve small batch at the end of the array
            whiteBatch = self.whites[startIndex:]
            blackBatch = self.blacks[startIndex:]

        # Generate labels
        whiteLabels = np.ones((whiteBatch.shape[0],))
        blackLabels = np.zeros((blackBatch.shape[0],))

        #Create arrays for batchs
        x = np.stack([whiteBatch, blackBatch], axis = 1)
        labels = np.stack([whiteLabels, blackLabels], axis = 1)

        # Randomly switch white and black board
        # Randomly generate array of 1 and 0 of length batchSize
        # Switch each index that contains 1
        swapIndices = np.random.randint(2, size = x.shape[0])
        x[swapIndices == 1] = np.flip(x[swapIndices == 1], axis = 1)
        labels[swapIndices == 1] = np.flip(labels[swapIndices == 1], axis = 1)

        # Split into two numpy arrays to pass into model
        leftBatch, rightBatch = np.split(x, 2, axis = 1)

        leftBatch = np.squeeze(leftBatch)
        rightBatch = np.squeeze(rightBatch)

        return [leftBatch, rightBatch], labels

    # Shuffle the order of the white and blacks
    def on_epoch_end(self):
        print("DATA GENERATOR: Shuffled white and black boards")
        np.random.shuffle(self.whites)
        np.random.shuffle(self.blacks)



# List to store weights and biases from autoencoder
weights = []

print("------------ Initialize Data Generator ------------")
trainGenerator = DeepchessDataGenerator(256)

print("------------ Beginning training ------------")
# Load pos2Vec model
pos2Vec = load_model(pos2VecPath)

# Inputs to Pos2Vec networks
inputLeft = Input(shape=(773,))
inputRight = Input(shape=(773,))

# Outputs from siamese pos2Vec networks
pos2VecLeftOut = pos2Vec(inputLeft)
pos2VecRightOut = pos2Vec(inputRight)
pos2VecOut = Concatenate()([pos2VecLeftOut, pos2VecRightOut])

# Build deepchess network
deepchess1 = Dense(deepchessLayers[0], activation='relu')(pos2VecOut)
deepchess2 = Dense(deepchessLayers[1], activation='relu')(deepchess1)
deepchess3 = Dense(deepchessLayers[2], activation='relu')(deepchess2)
deepchessOut = Dense(deepchessLayers[3], activation='softmax')(deepchess3)

# Compile model
deepchess = Model([inputLeft, inputRight], deepchessOut)
deepchess.compile(optimizer = 'adam', loss='categorical_crossentropy', metrics = ['acc'])

deepchess.fit_generator(trainGenerator, epochs = 50)

deepchess.save("./data/DeepchessModel.h5")















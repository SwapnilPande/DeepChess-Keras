import chess.pgn
import chess
import random
import numpy as np

# Open data
dataPath = "data/games.pgn"

numWhiteMoves = 1000000
numBlackMoves = 1000000


def getValidMoves(game):
    validMoves = []

    # Iterate over moves in game
    for i, move in enumerate(game.mainline_moves()):
        # Filter out first five moves and captures
        # These are filtered according to the methodology presented in the deepchess paper
        if(not game.board().is_capture(move) and (i >= 5)):
            # Append the move index to the validMoves list
            validMoves.append(i)

    return validMoves

def getBitboard(board):
    """
        A bitboard is a representation of the current board state
        There are a total of 64 squares on the board, 6 pieces, and 2 colors
        Each unique piece/color has 64 indices, with a 1 indicating that the piece exists at that location
        4 extra indices are for castling rights on each size
        1 extra index indicates whose turn it is
    """
    bitboard = np.zeros(2*6*64  + 5)

    pieceIndices = {
        'p': 0,
        'n': 1,
        'b': 2,
        'r': 3,
        'q': 4,
        'k': 5}

    for i in range(64):
        if board.piece_at(i):
            color = int(board.piece_at(i).color)
            bitboard[(6*color + pieceIndices[board.piece_at(i).symbol().lower()] + 12*i)] = 1

    bitboard[-1] = int(board.turn)
    bitboard[-2] = int(board.has_kingside_castling_rights(True))
    bitboard[-3] = int(board.has_kingside_castling_rights(False))
    bitboard[-4] = int(board.has_queenside_castling_rights(True))
    bitboard[-5] = int(board.has_queenside_castling_rights(False))

    return bitboard

def getBitboards(game, selectedMoves):


    return bitboards

# Adds 10 moves from game to moveArray at location moveIndex
def addMoves(game, moveArray, moveIndex):
    # Retrieve all vandidates for valid moves from the game
    # Candidates are moves that are not the first 5 and are not captures
    validMoves = getValidMoves(game)

    # List to store 10 randomly selected moves
    selectedMoves = []
    for i in range(10):
        if(not validMoves):
            break

        # Select move randomly, remove from valid moves
        move = random.choice(validMoves)
        validMoves.remove(move)
        selectedMoves.append(move)

    #print(selectedMoves)

    # Instantiate a new chess board
    board = chess.Board()
    moveCount = 0
    for i, move in enumerate(game.mainline_moves()):
        # Push new move to board
        board.push(move)

        # Break if maximum number of moves already reached
        if(moveIndex >= moveArray.shape[0]):
            break

        # Check if the current move is one of the selected moves
        if(i in selectedMoves):
            moveArray[moveIndex] = getBitboard(board)
            moveIndex += 1

    return moveIndex

# iterateOverData
# Iterates over the provided pgn file and extracts 10 random moves.
# The data is stored in numpy arrays
# Continues iterating until end of file or until the desired number of boards for each color win has been reached
def iterateOverData():
    # Initialize numpy arrays to store white and black moves
    whiteMoves = np.zeros((numWhiteMoves, 2*6*64  + 5))
    blackMoves = np.zeros((numBlackMoves, 2*6*64  + 5))

    # White and black move counts store how many white and black moves have been stored
    whiteMoveIndex = 0
    blackMoveIndex = 0
    count = 0

    # Openfile containing chess game data
    pgn = open(dataPath)

    # Loop over games in file
    while True:
        # Debug printing
        if(count % 1000 == 0):
            print("Game Number: {count}\tWhite Moves: {whiteMoves}\tBlack Moves: {blackMoves}".format(
                count = count,
                blackMoves = blackMoveIndex,
                whiteMoves = whiteMoveIndex))

        # Read in a single game from file
        game = chess.pgn.read_game(pgn)

        # Exit if end of file reached or if desired number of moves reached
        if((not game) or (whiteMoveIndex >= numWhiteMoves and blackMoveIndex >= numBlackMoves)):
            break
        if(game.headers["Result"] == "1-0" and whiteMoveIndex < numWhiteMoves):
            #print("Adding white game")
            whiteMoveIndex = addMoves(game, whiteMoves, whiteMoveIndex)
        if(game.headers["Result"] == "0-1" and blackMoveIndex < numBlackMoves):
            #print("adding black game")
            blackMoveIndex = addMoves(game, blackMoves, blackMoveIndex)

        #print(str(whiteMoveIndex) + "\t" + str(blackMoveIndex))

        count += 1

    return whiteMoves, blackMoves

white, black = iterateOverData()


np.save("data/white.npy", white)
np.save("data/black.npy", black)






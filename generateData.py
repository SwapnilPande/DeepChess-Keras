import chess.pgn
import chess
import random

# Open data
dataPath = "data/games.pgn"

numWhiteMoves = 1000000
numBlackMoves = 1000000


def getValidMoves(node):
    validMoves = []
    moveCount = 0

    while not node.is_end():
        # Retrieve next move
        nextNode = game.variation(0)

        # Retrieve move
        move = (game.board().san(nextNode.move))
        moveCount += 1

        # Filter out first five moves and captures
        if(("x" not in move) and (moveCount > 5):
            validMoves.append(moveCount)

        game = nextNode

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
            color = int(board.piece_at(i).color) + 1
            bitboard[(piece_idx[board.piece_at(i).symbol().lower()] + i * 6) * color] = 1

    bitboard[-1] = int(board.turn)
    bitboard[-2] = int(board.has_kingside_castling_rights(True))
    bitboard[-3] = int(board.has_kingside_castling_rights(False))
    bitboard[-4] = int(board.has_queenside_castling_rights(True))
    bitboard[-5] = int(board.has_queenside_castling_rights(False))

    return bitboard

def getBitboards(game, selectedMoves):
    # Instantiate a new chess board
    board = chess.Board()
    moveCount = 1

    bitboards = []

    for move in game.main_line():
        # Push new move to board
        board.push(move)

        if moveCount in selectedMoves:
            bitboards.append(getBitboard(board))


        moveCount += 1

def addMoves(game):
    # Retrieve all vandidates for valid moves from the game
    # Candidates are moves that are not the first 5 and are not captures
    validMoves = getValidMoves(game)

    # List to store 10 randomly selected moves
    selectedMoves = []
    for i in range(10):
        if(not validMoves):
            break

        # Select move
        move = random.choice(validMoves)
        validMoves.remove(move)
        selectedMoves.append(move)

    bitboards = getBitboards(game, selectedMoves)

    return bitboards


def iterateOverData(pgn):
    whiteMoves = []
    blackMoves = []

    # Openfile containing chess game data
    pgn = open(dataPath)

    # Add stopping condition
    while True:
        # Read in a single game from file
        game = chess.pgn.read_game(pgn)

        # Exit if end of file reached
        if not game:
            break

        if(game.headers["Result"] == "1-0" and len(whiteMoves) < numWhiteMoves):
            whiteMoves.extend(addMoves(game))
        if(game.headers["Result"] == "0-1" and len(blackMoves) < numBlackMoves):
            blackMoves.extend(addMoves(game))



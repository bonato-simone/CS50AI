"""
Tic Tac Toe Player
"""

from copy import deepcopy

X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """
    flat_board = [element for row in board for element in row]
    x = flat_board.count(X)
    o = flat_board.count(O)
    return X if (x + o) % 2 == 0 else O


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    actions_set = set()
    for row in range(3):
        for col in range(3):
            if board[row][col] == EMPTY:
                actions_set.add((row, col))
    return actions_set


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    i, j = action
    if board[i][j] != EMPTY:
        raise NameError("Cell (" + str(i) + ", " + str(j) + ") is already occupied")
    new_board = deepcopy(board)
    new_board[i][j] = player(board)
    return new_board


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    lines = []
    for i in range(3):
        lines.append([(i, 0), (i, 1), (i, 2)])
        lines.append([(0, i), (1, i), (2, i)])
    lines.append([(0, 0), (1, 1), (2, 2)])
    lines.append([(2, 0), (1, 1), (0, 2)])

    for line in lines:
        three_cells = []
        for row, col in line:
            three_cells.append(board[row][col])
        if three_cells[0] is not EMPTY and three_cells[0] == three_cells[1] == three_cells[2]:
            return three_cells[0]
    return None


def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    if winner(board) is not None:
        return True
    for row in range(3):
        for col in range(3):
            if board[row][col] == EMPTY:
                return False
    return True

def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    res = winner(board)
    if res == X:
        return 1
    elif res == O:
        return -1
    else:
        return 0


def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    if terminal(board):
        return None
    return max_value(board)[0] if player(board) == X else min_value(board)[0]


def max_value(board):
    """
    Returns the action with the higher value associated and the said value
    """
    optimalAction = None
    value = -2

    for action in actions(board):
        new_board = result(board, action)
        if terminal(new_board):
            new_value = utility(new_board)
        else:
            _, new_value = min_value(new_board)
        if new_value > value:
            optimalAction = action
            value = new_value
    return (optimalAction, value)


def min_value(board):
    """
    Returns the action with the lowest value associated and the said value
    """
    optimalAction = None
    value = 2

    for action in actions(board):
        new_board = result(board, action)
        if terminal(new_board):
            new_value = utility(new_board)
        else:
            _, new_value = max_value(new_board)
        if new_value < value:
            optimalAction = action
            value = new_value
    return (optimalAction, value)

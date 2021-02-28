board = [
    [7, 8, 0, 4, 0, 0, 1, 2, 0],
    [6, 0, 0, 0, 7, 5, 0, 0, 9],
    [0, 0, 0, 6, 0, 1, 0, 7, 8],
    [0, 0, 7, 0, 4, 0, 2, 6, 0],
    [0, 0, 1, 0, 5, 0, 9, 3, 0],
    [9, 0, 4, 0, 6, 0, 0, 0, 5],
    [0, 7, 0, 3, 0, 0, 0, 1, 2],
    [1, 2, 0, 0, 0, 7, 4, 0, 0],
    [0, 4, 9, 2, 0, 6, 0, 0, 7],
]

VALUES = range(1, 10)


def is_valid_collection(collection):
    non_zeros = list(filter(lambda digit: digit != 0, collection))
    n_non_zero_uniques = len(set(non_zeros))
    return len(non_zeros) == n_non_zero_uniques


def get_columns(board):
    dimensions = range(len(board))
    columns = [[] for _ in dimensions]
    for row_index in dimensions:
        for column_index in dimensions:
            columns[column_index].append(board[row_index][column_index])
    return columns


def get_squares(board):
    dimensions = range(len(board))
    squares = [[] for _ in dimensions]
    for row_index in dimensions:
        for column_index in dimensions:
            square_index = row_index // 3 * 3 + column_index // 3
            squares[square_index].append(board[row_index][column_index])
    return squares


def is_valid_board(board):
    has_valid_rows = all(map(is_valid_collection, board))
    if not has_valid_rows:
        return False
    columns = get_columns(board)
    has_valid_columns = all(map(is_valid_collection, columns))
    if not has_valid_columns:
        return False
    squares = get_squares(board)
    has_valid_squares = all(map(is_valid_collection, squares))
    if not has_valid_squares:
        return False
    return True


def get_flatened_board(board):
    return [digit for row in board for digit in row]


def get_next_empty_cell_indices(board):
    for row_index in range(len(board)):
        for column_index in range(len(board)):
            if board[row_index][column_index] == 0:
                return (row_index, column_index)
    return None


def solve(board):
    next_empty_indices = get_next_empty_cell_indices(board)
    if next_empty_indeces is None:
        print("Solved!")
        return board
    row_index, column_index = next_empty_indices
    for candidate in VALUES:
        board[row_index][column_index] = candidate
        if not is_valid_board(board):
            continue
        solved_board = solve(board)
        if solved_board is not None:
            return solved_board
    board[row_index][column_index] = 0

    
for row in solve(board):
  print(row)

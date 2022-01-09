from typing import List

import pulp

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

INDICES = range(0, 9)
VALUES = range(1, 10)


def get_objective_function():
    return 0


def get_board_variables():
    return pulp.LpVariable.dicts(
        "cell_values", (INDICES, INDICES, VALUES), cat="Binary"
    )


def add_constraints_to_lp(
    lp: pulp.LpProblem, constraints: List[pulp.LpVariable]
) -> None:
    for constraint in constraints:
        lp += constraint


def get_cell_value_constraints(indicators):
    """Ensure that every cell has exactly one value from VALUES."""
    constraints = [
        pulp.lpSum([indicators[row_index][column_index][value] for value in VALUES])
        == 1
        for row_index in INDICES
        for column_index in INDICES
    ]
    return constraints


def get_row_constraints(indicators):
    """Ensure that every row has no value more than once."""
    constraints = [
        pulp.lpSum(
            [indicators[row_index][column_index][value] for column_index in INDICES]
        )
        == 1
        for value in VALUES
        for row_index in INDICES
    ]
    return constraints


def get_column_constraints(indicators):
    """Ensure that every column has no value more than once."""
    constraints = [
        pulp.lpSum(
            [indicators[row_index][column_index][value] for row_index in INDICES]
        )
        == 1
        for value in VALUES
        for column_index in INDICES
    ]
    return constraints


def get_row_column_indices(square_index):
    start_row_index = (square_index // 3) * 3
    start_column_index = (square_index % 3) * 3
    return [
        (row_index, column_index)
        for row_index in range(start_row_index, start_row_index + 3)
        for column_index in range(start_column_index, start_column_index + 3)
    ]


def get_square_constraints(indicators):
    """Ensure that each of the 9 designated squares contains no value
    more than once."""
    constraints = [
        pulp.lpSum(
            indicators[row_index][column_index][value]
            for row_index, column_index in get_row_column_indices(square_index)
        )
        == 1
        for value in VALUES
        for square_index in INDICES
    ]
    return constraints


def get_starting_constraints(indicators, board):
    """Ensure the starting state of the board is not altered."""
    constraints = [
        indicators[row_index][column_index][board[row_index][column_index]] == 1
        for row_index in INDICES
        for column_index in INDICES
        if board[row_index][column_index] != 0
    ]
    return constraints


def insert_solution(board, indicators):
    for row_index in INDICES:
        for column_index in INDICES:
            for value in VALUES:
                if pulp.value(indicators[row_index][column_index][value]) != 1:
                    continue
                board[row_index][column_index] = value


def main():
    lp = pulp.LpProblem("sudoku")

    lp += get_objective_function()

    indicators = get_board_variables()

    add_constraints_to_lp(lp, get_cell_value_constraints(indicators))
    add_constraints_to_lp(lp, get_row_constraints(indicators))
    add_constraints_to_lp(lp, get_column_constraints(indicators))
    add_constraints_to_lp(lp, get_square_constraints(indicators))
    add_constraints_to_lp(lp, get_starting_constraints(indicators, board))

    lp.solve()

    insert_solution(board, indicators)

    for row in board:
        print(row)


if __name__ == "__main__":
    main()

from functools import cache, reduce
from itertools import chain

import gym
import numpy as np


@cache
def valid_board(n_blocks: int) -> np.ndarray:
    size = n_blocks ** 2
    rows = [
        [(i + offset) % (size) for i in range(0, size)] for offset in range(0, size)
    ]

    board = np.array(rows, dtype=np.uint8)
    indeces = np.fromiter(
        reduce(
            chain,
            map(
                lambda block_index: range(
                    block_index, block_index + ((n_blocks - 1) * n_blocks + 1), n_blocks
                ),
                range(n_blocks),
            ),
        ),
        dtype=np.uint8,
    )
    reordered_board = board[indeces, :]
    return reordered_board + 1


def random_board(n_blocks: int) -> np.ndarray:
    board = valid_board(n_blocks)
    new_board = board.copy()

    def permute_indeces(indeces: np.ndarray) -> np.ndarray:
        permutation = np.random.permutation(range(len(indeces)))
        return indeces[permutation]

    # Either permute rows or columns.
    permute_rows = np.random.binomial(1, 0.5)

    if permute_rows:
        # Shuffling row-blocks.
        original_indeces = np.array([i * n_blocks for i in range(n_blocks)])
        permuted_indeces = permute_indeces(original_indeces)
        for original_index, permuted_index in zip(original_indeces, permuted_indeces):
            new_board[original_index : original_index + n_blocks, :] = board[
                permuted_index : permuted_index + n_blocks, :
            ]
        # Shuffling rows within blocks.
        for row_block in range(n_blocks):
            original_indeces = np.arange(
                row_block * n_blocks, row_block * n_blocks + n_blocks
            )
            permuted_indeces = permute_indeces(original_indeces)
            new_board[original_indeces, :] = new_board[permuted_indeces, :]
    else:
        # Shuffling column-blocks.
        original_indeces = np.array([i * n_blocks for i in range(n_blocks)])
        permuted_indeces = permute_indeces(original_indeces)
        for original_index, permuted_index in zip(original_indeces, permuted_indeces):
            new_board[:, original_index : original_index + n_blocks] = board[
                :, permuted_index : permuted_index + n_blocks
            ]
        # Shuffling columns within blocks.
        for column_block in range(n_blocks):
            original_indeces = np.arange(
                column_block * n_blocks, column_block * n_blocks + n_blocks
            )
            permuted_indeces = permute_indeces(original_indeces)
            new_board[:, original_indeces] = new_board[:, permuted_indeces]
    return new_board


def almost_solved_board(solved_board: np.ndarray) -> np.ndarray:
    board_copy = solved_board.copy()
    indeces = np.random.randint(0, solved_board.shape[0], 2)
    board_copy[indeces[0], indeces[1]] = 0
    return board_copy


def hole_indeces(board: np.ndarray) -> tuple[int, int]:
    for x_index in range(len(board)):
        for y_index in range(len(board)):
            if board[x_index][y_index] == 0:
                return (x_index, y_index)
    raise ValueError("No hole in board.")


def compare(array_a: np.ndarray, array_b: np.ndarray) -> tuple[bool, int]:
    if np.array_equal(array_a, array_b):
        return True, 1
    return False, 0


class SudokuEnv(gym.Env):
    def __init__(self, n_blocks: int):
        self.n_blocks = n_blocks
        size = n_blocks ** 2
        # print(size)
        self.observation_space = gym.spaces.Box(
            low=0, high=size, shape=(size, size), dtype=np.uint8
        )
        # The start keyword is not supported in gym version 0.21.0
        # self.action_space = gym.spaces.Discrete(size, start=1)
        self.action_space = gym.spaces.Discrete(size)
        self._setup_boards()
        self.n_steps = 0
        self.empirical_measurement_plan = {action: 0 for action in range(1, size + 1)}

    def _setup_boards(self):
        self.board_solved = random_board(self.n_blocks)
        self.initial_board = almost_solved_board(self.board_solved)
        self.board = self.initial_board.copy()
        self.hole_indeces = hole_indeces(self.initial_board)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        # See comment in `__init__`.
        action += 1
        self.empirical_measurement_plan[action] += 1
        self.n_steps += 1
        x_index, y_index = self.hole_indeces
        self.board[x_index, y_index] = action
        done, reward = compare(self.board, self.board_solved)
        if not done:
            self.board[x_index, y_index] = 0
        return (
            self.board,
            float(reward),
            done,
            {
                "n_steps": self.n_steps,
                "empirical_measurement_plan": self.empirical_measurement_plan,
            },
        )

    def reset(self) -> np.ndarray:
        self._setup_boards()
        self._n_steps = 0
        return self.board

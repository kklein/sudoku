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


def permute_indeces(indeces: np.ndarray) -> np.ndarray:
    permutation = np.random.permutation(range(len(indeces)))
    return indeces[permutation]


def permute_blocks(
    old_board: np.ndarray, new_board: np.ndarray, n_blocks: int, axis: int
) -> None:
    original_indeces = np.array([i * n_blocks for i in range(n_blocks)])
    permuted_indeces = permute_indeces(original_indeces)
    all_slice = slice(None)
    for original_index, permuted_index in zip(original_indeces, permuted_indeces):
        original_slice = slice(original_index, original_index + n_blocks)
        permuted_slice = slice(permuted_index, permuted_index + n_blocks)
        if axis == 0:
            original_slicing = (original_slice, all_slice)
            permuted_slicing = (permuted_slice, all_slice)
        else:
            original_slicing = (all_slice, original_slice)
            permuted_slicing = (all_slice, permuted_slice)
        new_board[original_slicing] = old_board[permuted_slicing]


def permute_within_block_vectors(board: np.ndarray, n_blocks: int, axis: int) -> None:
    all_slice = slice(None)
    for block_index in range(n_blocks):
        original_indeces = np.arange(
            block_index * n_blocks, block_index * n_blocks + n_blocks
        )
        permuted_indeces = permute_indeces(original_indeces)
        if axis == 0:
            original_slicing = (original_indeces, all_slice)
            permuted_slicing = (permuted_indeces, all_slice)
        else:
            original_slicing = (all_slice, original_indeces)
            permuted_slicing = (all_slice, permuted_indeces)
        board[original_slicing] = board[permuted_slicing]


def random_board(n_blocks: int) -> np.ndarray:
    old_board = valid_board(n_blocks)
    new_board = old_board.copy()

    # Either permute rows or columns. If 1, permute rows.
    axis = np.random.binomial(1, 0.5)

    permute_blocks(old_board, new_board, n_blocks, axis=axis)
    permute_within_block_vectors(new_board, n_blocks, axis=axis)

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

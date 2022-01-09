from stable_baselines3 import DQN

from sudoku_env import SudokuEnv

N_BLOCKS = 3
N_TIMESTEPS = 1_000_000


def print_samples(model, n_samples: int = 10):
    # Need to instantiate a new environment here in oder to have a reset
    # measurement plan.
    env = SudokuEnv(n_blocks=N_BLOCKS)
    obs = env.board
    for _ in range(n_samples):
        print(obs)
        action, _states = model.predict(obs, deterministic=True)
        obs, _, done, _ = env.step(action)
        print(action + 1)
        if done:
            print("HIT!")
        else:
            print("MISS :S")
        print("----")
        obs = env.reset()


def print_stats(model, n_samples: int = 1000):
    env = SudokuEnv(n_blocks=N_BLOCKS)
    obs = env.board
    n_solved = 0
    for _ in range(n_samples):
        action, _states = model.predict(obs, deterministic=True)
        obs, _, done, _ = env.step(action)
        if done:
            n_solved += 1
        obs = env.reset()
    print(f"Accuracy: {n_solved / n_samples}")
    print(f"Empirical measurement plan on samples: {env.empirical_measurement_plan}")


def main():
    file_name = f"sudoku_{N_BLOCKS}_{N_TIMESTEPS/1_000_000}M"
    loaded_model = DQN.load(file_name)
    print_samples(loaded_model)
    print_stats(loaded_model)


if __name__ == "__main__":
    main()

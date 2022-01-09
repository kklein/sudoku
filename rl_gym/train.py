from stable_baselines3 import DQN

from sudoku_env import SudokuEnv

N_BLOCKS = 3
N_TIMESTEPS = 1_000_000

env = SudokuEnv(n_blocks=N_BLOCKS)

trained_model = DQN("MlpPolicy", env, verbose=1)

file_name = f"sudoku_{N_BLOCKS}_{N_TIMESTEPS}"

trained_model.learn(total_timesteps=N_TIMESTEPS)
trained_model.save(file_name)
print(f"Model training finished. Model saved as {file_name}.zip.")
print(f"Empirical measurement plan of saved model: {env.empirical_measurement_plan}")

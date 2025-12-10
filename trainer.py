import gymnasium as gym
from stable_baselines3 import DQN
from autodriving2d.envs import CityDrive
from stable_baselines3.common.utils import set_random_seed
import torch
from tqdm import tqdm

import numpy as np


# Create environment
def make_env():
    """
    Utility function for multiprocessed env.
    """

    def _init():
        env = CityDrive(render_mode=None)  # Must be None for multiprocessing
        return env

    return _init


# Create environment
if __name__ == "__main__":
    num_subprocesses = 4  # Adjust based on your CPU cores

    env = CityDrive(render_mode=None, continuous=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # DQN model
    model = DQN(
        policy="MultiInputPolicy",
        env=env,
        learning_rate=1e-4,
        buffer_size=100000,
        learning_starts=1000,
        batch_size=64,
        gamma=0.99,
        train_freq=4,
        target_update_interval=1000,
        verbose=1,
        exploration_fraction=0.1,
        exploration_final_eps=0.05,
        device=device,
    )

# Train
print("trainin..")
model.learn(total_timesteps=300000, log_interval=10, progress_bar=True)
model.save("dqn_citydrive")
env.close()

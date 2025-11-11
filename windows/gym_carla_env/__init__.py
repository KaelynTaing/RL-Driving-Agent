from gymnasium.envs.registration import register

register(
    id="CarlaEnv-v0",
    entry_point="gym_carla_env.carla_env:CarlaEnv",
)
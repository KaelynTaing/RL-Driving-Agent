from gymnasium.envs.registration import register

register(
    id="autodriving2d/GridWorld-v0",
    entry_point="autodriving2d.envs:GridWorldEnv",
)

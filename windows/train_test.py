import gymnasium as gym
import gym_carla_env
import cv2
import os

def main():
    # Make output directory if it doesn't exist
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    env = gym.make("CarlaEnv-v0")

    obs, info = env.reset()
    print("Initial observation shape:", obs.shape)

    for t in range(10000):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        print(f"Step {t}: reward={reward}, done={done}")

        # Save observation as image
        filename = os.path.join(output_dir, f"frame_{t:03d}.png")
        if(t % 500 == 0 or t == 0):
            cv2.imwrite(filename, obs)

        if done:
            break

    env.close()
    print("Frames saved to output/ folder.")

if __name__ == "__main__":
    main()

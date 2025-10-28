import gymnasium as gym
import ale_py
import numpy as np
import torch
from env_utils import GrayScaleObservation, ResizeObservation, FrameStack
from agent import PPOAgent
from rnd import RNDModule
import wandb
from storage import RolloutStorage

# Set up the Gymnasium env
def make_env(env_id, render_mode=None):
    gym.register_envs(ale_py)
    env = gym.make(env_id, render_mode=render_mode)
    # Game screen frame preprocessing wrappers
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, (84, 84))
    env = FrameStack(env, 4)
    return env

# --- Hyperparameters ---
# Can either be pong or montezuma
# MontezumaRevenge-v5 = sparse rewards, good for testing RND
# Pong-v5 = dense rewards, baseline for PPO
ENV_ID = "ALE/MontezumaRevenge-v5"
USE_RND = True # False: BASELINE PPO; True: PPO + RND
NUM_STEPS_PER_ROLLOUT = 2048
LEARNING_RATE = 2.5e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPSILON = 0.1
PPO_EPOCHS = 4
NUM_MINI_BATCHES = 4
VALUE_LOSS_COEF = 0.5
ENTROPY_COEF = 0.01

# RND Hyperparameters (only used if USE_RND is True)
INT_REWARD_COEF = 1.0 
RND_LEARNING_RATE = 1e-4

# Total training timesteps
TOTAL_TIMESTEPS = 1_000_000

def main():
    # Set run name
    run_name = f"PPO_RND" if USE_RND else f"PPO_Baseline"

    # Set up the WanDB logging
    wandb.init(
        project="cs395t-ppo-rnd",
        name=run_name,
        config={
            "env_id": ENV_ID, "total_timesteps": TOTAL_TIMESTEPS,
            "use_rnd": USE_RND,
            "num_steps_per_rollout": NUM_STEPS_PER_ROLLOUT,
            "learning_rate": LEARNING_RATE, "gamma": GAMMA,
            "gae_lambda": GAE_LAMBDA, "clip_epsilon": CLIP_EPSILON,
            "ppo_epochs": PPO_EPOCHS, "num_mini_batches": NUM_MINI_BATCHES,
            "value_loss_coef": VALUE_LOSS_COEF, "entropy_coef": ENTROPY_COEF,
            "int_reward_coef": INT_REWARD_COEF if USE_RND else 0,
            "rnd_learning_rate": RND_LEARNING_RATE if USE_RND else 0
        }
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = make_env(ENV_ID, render_mode=None)
    num_actions = env.action_space.n
    obs_shape = env.observation_space.shape
    storage_obs_shape = (obs_shape[2], obs_shape[0], obs_shape[1]) 

    agent = PPOAgent(
        num_actions,
        LEARNING_RATE,
        GAMMA,
        GAE_LAMBDA,
        CLIP_EPSILON,
        VALUE_LOSS_COEF,
        ENTROPY_COEF,
        PPO_EPOCHS,
        NUM_MINI_BATCHES,
        INT_REWARD_COEF if USE_RND else 0.0 # 0 if RND is off
    )

    storage = RolloutStorage(NUM_STEPS_PER_ROLLOUT, storage_obs_shape, device)

    # Initialize RND Module if enabled
    rnd_module = None
    if USE_RND:
        rnd_module = RNDModule(device, learning_rate=RND_LEARNING_RATE)
        print("RND Module Initialized.")

    num_updates = TOTAL_TIMESTEPS // NUM_STEPS_PER_ROLLOUT
    obs, _ = env.reset(seed=42)
    obs = torch.tensor(np.transpose(obs, (2, 0, 1)), dtype=torch.float32).to(device)
    storage.reset()
    storage.obs[0].copy_(obs)
    total_timesteps = 0
    # BEGIN TRAINING LOOP
    for update in range(1, num_updates + 1):
        ep_ext_rewards = []
        ep_int_rewards = []
        current_ep_ext_reward = 0
        current_ep_int_reward = 0

        # Step 1: Rollout Collection
        for step in range(NUM_STEPS_PER_ROLLOUT):
            total_timesteps += 1
            current_obs_chw = obs
            action, log_prob, value = agent.get_action_and_value(current_obs_chw)

            int_reward = 0.0 # Default to 0
            if USE_RND:
                int_reward = rnd_module.compute_intrinsic_reward(current_obs_chw)

            next_obs_hwc, ext_reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            next_obs_chw = torch.tensor(np.transpose(next_obs_hwc, (2, 0, 1)), dtype=torch.float32).to(device)

            storage.insert(
                obs=current_obs_chw,
                action=torch.tensor(action, dtype=torch.long).to(device),
                log_prob=log_prob,
                value=value,
                reward=torch.tensor(ext_reward, dtype=torch.float32).to(device),
                int_reward=torch.tensor(int_reward, dtype=torch.float32).to(device),
                done=torch.tensor(done, dtype=torch.float32).to(device)
            )

            obs = next_obs_chw

            current_ep_ext_reward += ext_reward
            current_ep_int_reward += int_reward

            if done:
                ep_ext_rewards.append(current_ep_ext_reward)
                ep_int_rewards.append(current_ep_int_reward)
                current_ep_ext_reward = 0
                current_ep_int_reward = 0

                obs, info = env.reset()
                obs = torch.tensor(np.transpose(obs, (2, 0, 1)), dtype=torch.float32).to(device)

        # End of Rollout
        with torch.no_grad():
            _, _, last_value = agent.get_action_and_value(obs)

        storage.store_last(obs, last_value)

        # RND Training
        rnd_loss = 0
        if USE_RND:
            all_obs = storage.get_data()[0][:-1] 
            rnd_loss = rnd_module.train(all_obs)

        # 3. PPO Training
        agent.train(storage)

        # 4. Logging
        log_data = {
            "timestep": total_timesteps,
            "update": update,
            "mean_episode_ext_reward": np.mean(ep_ext_rewards) if ep_ext_rewards else 0,
        }
        if USE_RND:
            log_data["rnd_loss"] = rnd_loss
            log_data["mean_episode_int_reward"] = np.mean(ep_int_rewards) if ep_int_rewards else 0

        wandb.log(log_data)

        # Reset storage for next rollout
        storage.reset()
        storage.obs[0].copy_(obs)

        print(f"Update {update}/{num_updates} complete. Timesteps: {total_timesteps}")
        if ep_ext_rewards:
             print(f"  Mean Extrinsic Reward: {np.mean(ep_ext_rewards):.2f}")

    env.close()
    wandb.finish()

if __name__ == "__main__":
    main()

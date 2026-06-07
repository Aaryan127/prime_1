
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback



ENV_ID      = "CartPole-v1"
TOTAL_STEPS = 200_000
N_ENVS      = 4
LR          = 3e-4
N_STEPS     = 2048
BATCH_SIZE  = 64
N_EPOCHS    = 10
GAMMA       = 0.99
GAE_LAMBDA  = 0.95
CLIP_RANGE  = 0.2
VF_COEF     = 0.5
ENT_COEF    = 0.01


class RewardLogger(BaseCallback):
    def __init__(self):
        super().__init__()
        self.ep_rewards = []

    def _on_step(self):
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.ep_rewards.append((self.num_timesteps, info["episode"]["r"]))
        return True

# Train


def train():
    env      = make_vec_env(ENV_ID, n_envs=N_ENVS)
    callback = RewardLogger()

    model = PPO(
        policy        = "MlpPolicy",
        env           = env,
        learning_rate = LR,
        n_steps       = N_STEPS,
        batch_size    = BATCH_SIZE,
        n_epochs      = N_EPOCHS,
        gamma         = GAMMA,
        gae_lambda    = GAE_LAMBDA,
        clip_range    = CLIP_RANGE,
        vf_coef       = VF_COEF,
        ent_coef      = ENT_COEF,
        verbose       = 0,
    )

    model.learn(total_timesteps=TOTAL_STEPS, callback=callback)
    model.save("ppo_cartpole")
    env.close()

    return model, callback.ep_rewards


def evaluate(model, n_episodes=10):
    import gymnasium as gym
    env     = gym.make(ENV_ID)
    rewards = []

    for _ in range(n_episodes):
        obs, _  = env.reset()
        total   = 0.0
        done    = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done   = terminated or truncated
            total += reward
        rewards.append(total)

    env.close()
    print(f"Mean reward: {np.mean(rewards):.1f} +/- {np.std(rewards):.1f}")



def plot(ep_rewards):
    steps, rewards = zip(*ep_rewards)

    plt.figure(figsize=(8, 4))
    plt.plot(steps, rewards, alpha=0.3, label="Episode reward")
    plt.xlabel("Timestep")
    plt.ylabel("Reward")
    plt.title("PPO on CartPole-v1")
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.close()



if __name__ == "__main__":
    model, ep_rewards = train()
    evaluate(model)
    plot(ep_rewards)
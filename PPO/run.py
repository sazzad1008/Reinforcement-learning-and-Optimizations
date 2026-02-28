from PPO.config import PPOConfig
from PPO.trainer import train_ppo


if __name__ == "__main__":
    result = train_ppo(PPOConfig())
    print({"train_accuracy": round(result.train_accuracy, 4), "test_accuracy": round(result.test_accuracy, 4)})

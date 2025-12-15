import torch

from training.selfPlay import SelfPlayTrainer
from training.ppo import PPOConfig
from evaluation.eval import run_head_to_head


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = PPOConfig()
    trainer = SelfPlayTrainer(device, config)
    for step in range(5):
        metrics = trainer.train_step(episodes=32)
        exploit = trainer.estimate_exploitability(br_episodes=16, br_updates=1)
        print("step {} policy_loss={:.4f} value_loss={:.4f} entropy={:.4f} exploit={:.4f}".format(step, metrics["policy_loss"], metrics["value_loss"], metrics["entropy"], exploit))
    ev = run_head_to_head(trainer.net, hands=200, device=device)
    print("ev vs baseline: {:.2f} bb/100".format(ev))


if __name__ == "__main__":
    main()

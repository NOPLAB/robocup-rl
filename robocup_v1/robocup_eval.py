from typing import Tuple
import genesis as gs
import numpy as np
from sb3_contrib import RecurrentPPO
import torch
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from sb3_contrib.common.recurrent.type_aliases import RNNStates

from robocup_env import RoboCupEnv, get_env_cfg


class OnnxableSB3Policy(torch.nn.Module):
    def __init__(self, policy: RecurrentActorCriticPolicy):
        super().__init__()
        self.policy = policy

    def forward(
        self,
        observation: torch.Tensor,
        pi: torch.Tensor,
        vf: torch.Tensor,
        episode_start: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.policy(
            observation,
            RNNStates(
                pi=pi,
                vf=vf,
            ),
            episode_start,
            deterministic=True,
        )


def main():
    # Load the model
    model = RecurrentPPO.load(path="checkpoints/ppo_robocup_step_9175040", device="cpu")

    observation_size = model.observation_space.shape
    dummy_observation = np.zeros((1, *observation_size), dtype=np.float32)
    dummy_episode_starts = np.ones((1,), dtype=bool)

    _, dummy_lstm_states = model.predict(
        observation=dummy_observation,
        state=None,
        episode_start=dummy_episode_starts,
        deterministic=True,
    )

    print(f"dummy_lstm_states: {dummy_lstm_states[0]}")

    torch.onnx.export(
        model=OnnxableSB3Policy(model.policy),
        args=(
            torch.randn((1, *observation_size), dtype=torch.float32),
            torch.from_numpy(np.array([dummy_lstm_states[0], dummy_lstm_states[1]])),
            torch.from_numpy(np.array([dummy_lstm_states[0], dummy_lstm_states[1]])),
            torch.ones((1,), dtype=torch.float32),
        ),
        f="checkpoints/robocup.onnx",
    )

    gs.init()

    env_cfg = get_env_cfg()

    env = RoboCupEnv(
        num_envs=1,
        env_cfg=env_cfg,
        episode_length_s=5,
        show_viewer=True,
    )

    lstm_states = None
    episode_starts = np.ones((1,), dtype=bool)

    obs = env.reset()
    for _ in range(1000):
        actions, lstm_states = model.predict(
            observation=obs[0],
            state=lstm_states,
            episode_start=episode_starts,
            deterministic=True,
        )
        obs, reward, done, _ = env.step(actions=np.array([actions]))

        episode_starts = done

        print(f"actions: {actions}")
        print(f"obs: {obs[0]}")
        print(f"reward: {reward[0]}")
        print(f"done: {done[0]}")

        if done:
            env.reset()


if __name__ == "__main__":
    main()

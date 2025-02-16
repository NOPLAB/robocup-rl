import genesis as gs

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import BaseCallback

from robocup_env import RoboCupEnv, get_env_cfg


class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)
        self._log_freq = 50

    def _on_step(self) -> bool:
        if self.n_calls % self._log_freq == 0:
            rewards = self.locals["env"].episode_reward_sums

            rewards_mean = sum(rewards) / len(rewards)

            self.logger.record("episode rewards_mean", rewards_mean)

        return True


class ModelCheckpointCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(ModelCheckpointCallback, self).__init__(verbose)
        self._log_freq = 64

    def _on_step(self) -> bool:
        if self.n_calls % self._log_freq == 0:
            self.model.save("checkpoints/ppo_robocup_step_" + str(self.num_timesteps))

        return True


def main():
    gs.init()

    env_cfg = get_env_cfg()

    env = RoboCupEnv(
        num_envs=4096,
        env_cfg=env_cfg,
        episode_length_s=5,
        show_viewer=False,
    )

    policy_kwargs = dict(net_arch=dict(pi=[128, 128, 128], vf=[128, 128, 128]))
    model = RecurrentPPO(
        "MlpLstmPolicy",
        env,
        n_steps=128,
        batch_size=128 * 512,
        verbose=1,
        device="cuda",
        tensorboard_log="logs/robocup",
        policy_kwargs=policy_kwargs,
    ).learn(
        total_timesteps=100000000,
        callback=[TensorboardCallback(), ModelCheckpointCallback()],
        progress_bar=True,
    )

    # Save the model
    model.save("checkpoints/ppo_robocup")


if __name__ == "__main__":
    main()

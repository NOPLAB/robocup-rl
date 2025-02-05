import genesis as gs

from robocup_env import RoboCupEnv, get_env_cfg


def main():
    gs.init()

    env_cfg = get_env_cfg()

    env = RoboCupEnv(
        num_envs=1,
        env_cfg=env_cfg,
        episode_length_s=10,
        show_viewer=True,
    )

    env.reset()

    # Wait for the user to press enter
    input("Press Enter to continue...")


if __name__ == "__main__":
    main()

import gym

from game_models.dqn_game_model import DQNTrainer, DQNSolver
from game_models.ge_game_model import GETrainer, GESolver

ENV_NAME = "Breakout-v0"


def atari():
    env = gym.make(ENV_NAME)

    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    game_model = DQNTrainer(ENV_NAME, observation_space, action_space)

    run = 0
    while True:
        run += 1
        state = env.reset()
        step = 0
        while True:
            step += 1
            env.render()
            action = env.action_space.sample()
            state_next, reward, terminal, info = env.step(action)
            if terminal:
                print "Run: " + str(run) + ", score: " + str(step)
                game_model.save_score(step)
                break


if __name__ == "__main__":
    atari()

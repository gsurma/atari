import gym

from scores.score_logger import ScoreLogger

ENV_NAME = "Breakout-v0"


def atari():
    env = gym.make(ENV_NAME)
    score_logger = ScoreLogger(ENV_NAME, "./scores/dqn/scores.csv", "./scores/dqn/scores.png")
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n

    run = 0
    while True:
        run += 1
        state = env.reset()
        step = 0
        while True:
            step += 1
            env.render()
            action = env.action_space.sample() #TODO:
            state_next, reward, terminal, info = env.step(action)
            if terminal:
                print "Run: " + str(run) + ", score: " + str(step)
                score_logger.add_score(step)
                break


if __name__ == "__main__":
    atari()

import gym
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
#np.set_printoptions(threshold=np.nan)

from game_models.dqn_game_model import DQNTrainer, DQNSolver
from game_models.ge_game_model import GETrainer, GESolver

GAME_NAME = "Breakout"
ENV_NAME = GAME_NAME + "Deterministic-v4"
FRAMES_IN_OBSERVATION = 4


def preprocess(frame):
    frame = cv2.cvtColor(cv2.resize(frame, (84, 110)), cv2.COLOR_BGR2GRAY)
    frame = frame[26:110, :]
    _, frame = cv2.threshold(frame, 1, 255, cv2.THRESH_BINARY)
    return np.reshape(frame, (84, 84, 1)) # Returning square grayscale frame

def atari(game_model, env):
    observation = deque(maxlen=FRAMES_IN_OBSERVATION)
    run = 0
    while True:
        run += 1
        state = env.reset()
        [observation.append(preprocess(state)) for _ in xrange(FRAMES_IN_OBSERVATION)]
        step = 0
        while True:
            step += 1
            env.render()
            action = game_model.move(state)
            state_next, reward, terminal, info = env.step(action)
            observation.append(preprocess(state_next))
            game_model.remember(state, action, reward, state_next, terminal)
            state = state_next
            if terminal:
                print "Run: " + str(run) + ", score: " + str(step)
                game_model.save_score(step)
                break
            game_model.experience_replay()


if __name__ == "__main__":
    env = gym.make(ENV_NAME)
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    game_model = DQNTrainer(GAME_NAME, observation_space, action_space)
    atari(game_model, env)

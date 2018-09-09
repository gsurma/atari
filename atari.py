import gym
import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt
from collections import deque
#np.set_printoptions(threshold=np.nan)

from game_models.dqn_game_model import DQNTrainer, DQNSolver
from game_models.ge_game_model import GETrainer, GESolver

FRAMES_IN_OBSERVATION = 4
OBSERVATION_SPACE = 84
WARMUP_STEPS = 1000
UPDATE_FREQUENCY = 4


def preprocess_frame(frame):
    frame = cv2.cvtColor(cv2.resize(frame, (84, 110)), cv2.COLOR_BGR2GRAY)
    frame = frame[26:110, :]
    _, frame = cv2.threshold(frame, 1, 255, cv2.THRESH_BINARY)
    return np.reshape(frame, (OBSERVATION_SPACE, OBSERVATION_SPACE, 1))

def prepare_observation(observation_queue):
    observation = np.array(observation_queue)
    observation = observation.reshape([OBSERVATION_SPACE, OBSERVATION_SPACE, FRAMES_IN_OBSERVATION])
    observation = np.expand_dims(observation, axis=0)
    return observation

def atari(game_model, env):
    run = 0
    total_step = 0
    while True:
        run += 1
        initial_state = env.reset()
        observation_queue = deque(maxlen=FRAMES_IN_OBSERVATION)
        [observation_queue.append(preprocess_frame(initial_state)) for _ in xrange(FRAMES_IN_OBSERVATION)]
        observation = prepare_observation(observation_queue)
        step = 0
        score = 0
        while True:
            total_step += 1
            step += 1
            env.render()

            action = game_model.move(observation)
            state_next, reward, terminal, info = env.step(action)
            reward = np.clip(reward, -1, 1)
            score += reward

            observation_queue.append(preprocess_frame(state_next))
            new_observation = prepare_observation(observation_queue)
            game_model.remember(observation, action, reward, new_observation, terminal)
            observation = new_observation

            if terminal:
                print "Run: " + str(run) + ", score: " + str(score) + ", steps: " + str(step) + ", total steps: " + str(total_step)
                if game_model.exploration_rate is not None:
                    print "Exploration rate: " + str(game_model.exploration_rate)
                game_model.save_score(int(score))
                game_model.save_run_duration(int(step))
                game_model.save_model()
                print ""
                break
            game_model.update_exploration_rate()
            if total_step > WARMUP_STEPS and total_step % UPDATE_FREQUENCY == 0:
                game_model.experience_replay()

def args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-gn", "--game_name", help="Choose from available games: Breakout, Pong, SpaceInvaders") #TODO list
    parser.add_argument("-m", "--mode", help="Choose from available modes: dqn_train, dqn_test, ge_train, ge_test")
    args = parser.parse_args()
    game_mode = "dqn_train" if args.mode is None else args.mode
    game_name = "Breakout" if args.game_name is None else args.game_name
    print "Selected game: " + str(game_name)
    print "Selected mode: " + str(game_mode)
    return game_name, game_mode

def game_model(game_mode,game_name, action_space):
    if game_mode == "dqn_train":
        return DQNTrainer(game_name, OBSERVATION_SPACE, action_space)
    elif game_mode == "dqn_test":
        return DQNSolver(game_name, OBSERVATION_SPACE, action_space)
    elif game_mode == "ge_train":
        return GETrainer(game_name, OBSERVATION_SPACE, action_space)
    elif game_mode == "ge_test":
        return GESolver(game_name, OBSERVATION_SPACE, action_space)
    else:
        print "Unrecognized mode. Use --help"
        exit(1)

if __name__ == "__main__":
    game_name, game_mode = args()
    env_name = game_name + "Deterministic-v4"  # It handles frame skipping (4) at every iteration
    env = gym.make(env_name)
    action_space = env.action_space.n
    atari(game_model(game_mode, game_name, action_space), env)

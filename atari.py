import gym
import numpy as np
import argparse
from PIL import Image
from game_models.dqn_game_model import DQNTrainer, DQNSolver
from game_models.ge_game_model import GETrainer, GESolver

FRAMES_IN_OBSERVATION = 4
OBSERVATION_SPACE = 84


class Atari:

    def __init__(self):
        game_name, game_mode = self._args()
        env_name = game_name + "Deterministic-v4" # It handles frame skipping (4) at every iteration
        env = gym.make(env_name)
        self._main_loop(self._game_model(game_mode, game_name, env.action_space.n), env)

    def _main_loop(self, game_model, env):
        run = 0
        total_step = 0
        while True:
            run += 1
            initial_state = env.reset()
            observation = self._preprocess_observation(initial_state)
            current_state = np.array([observation]*FRAMES_IN_OBSERVATION)

            step = 0
            score = 0
            while True:
                total_step += 1
                step += 1
                env.render()

                action = game_model.move(current_state)
                state, reward, terminal, info = env.step(action)
                reward = np.clip(reward, -1, 1)
                score += reward

                next_state = np.append(current_state[1:], [self._preprocess_observation(state)], axis=0)
                game_model.remember(current_state, action, reward, next_state, terminal)
                current_state = next_state

                if terminal:
                    print "Run: " + str(run) + ", score: " + str(score) + ", steps: " + str(step) + ", total steps: " + str(total_step)
                    game_model.save_run(score, step)
                    print ""
                    break
                game_model.step_update(total_step)

    def _preprocess_observation(self, obs):
        image = Image.fromarray(obs, "RGB").convert("L").resize((OBSERVATION_SPACE, OBSERVATION_SPACE))
        return np.asarray(image.getdata(), dtype=np.uint8).reshape(image.size[1], image.size[0])

    def _args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("-g", "--game", help="Choose from available games: Breakout, Pong, SpaceInvaders") #TODO list
        parser.add_argument("-m", "--mode", help="Choose from available modes: dqn_train, dqn_test, ge_train, ge_test")
        args = parser.parse_args()
        game_mode = "dqn_train" if args.mode is None else args.mode
        game_name = "Breakout" if args.game is None else args.game
        print "Selected game: " + str(game_name)
        print "Selected mode: " + str(game_mode)
        return game_name, game_mode

    def _game_model(self, game_mode,game_name, action_space):
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
    from tensorflow.python.client import device_lib
    print device_lib.list_local_devices()
    Atari()

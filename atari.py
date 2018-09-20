import gym
import numpy as np
import argparse
import atari_py
from PIL import Image
from game_models.ddqn_game_model import DDQNTrainer, DDQNSolver
from game_models.ge_game_model import GETrainer, GESolver

FRAMES_IN_OBSERVATION = 4
FRAME_SIZE = 84
INPUT_SHAPE = (FRAMES_IN_OBSERVATION, FRAME_SIZE, FRAME_SIZE)


class Atari:

    def __init__(self):
        game_name, game_mode, render, total_step_limit = self._args()
        env_name = game_name + "Deterministic-v4"  # Handles frame skipping (4) at every iteration
        env = gym.make(env_name)
        self._main_loop(self._game_model(game_mode, game_name, env.action_space.n), env, render, total_step_limit)

    def _main_loop(self, game_model, env, render, total_step_limit):
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
                if total_step >= total_step_limit:
                    print "Reached total step limit of: " + str(total_step_limit)
                    exit(0)
                total_step += 1
                step += 1

                if render:
                    env.render()

                action = game_model.move(current_state)
                state, reward, terminal, info = env.step(action)
                reward = np.clip(reward, -1, 1)
                score += reward
                observation = self._preprocess_observation(state)
                next_state = np.append(current_state[1:], [observation], axis=0)
                game_model.remember(current_state, action, reward, next_state, terminal)
                current_state = next_state

                game_model.step_update(total_step)

                if terminal:
                    game_model.save_run(score, step, run)
                    break

    def _preprocess_observation(self, obs):
        image = Image.fromarray(obs, "RGB").convert("L").resize((FRAME_SIZE, FRAME_SIZE))
        return np.asarray(image.getdata(), dtype=np.uint8).reshape(image.size[1], image.size[0]) #TODO: possibly memory heavy

    def _args(self):
        parser = argparse.ArgumentParser()
        available_games = list((''.join(x.capitalize() or '_' for x in word.split('_')) for word in atari_py.list_games()))
        parser.add_argument("-g", "--game", help="Choose from available games: " + str(available_games) + ". Default is 'breakout'.", default="SpaceInvaders")
        parser.add_argument("-m", "--mode", help="Choose from available modes: ddqn_train, ddqn_test, ge_train, ge_test. Default is 'ddqn_training'.", default="ddqn_training")
        parser.add_argument("-r", "--render", help="Choose if the game should be rendered. Default is 'False'.", default=False)
        parser.add_argument("-tsl", "--total_step_limit", help="Choose how many total steps (frames visible by agent) should be performed. Default is '10000000'.", default=10000000)
        args = parser.parse_args()
        game_mode = args.mode
        game_name = args.game
        render = args.render
        total_step_limit = args.total_step_limit
        print "Selected game: " + str(game_name)
        print "Selected mode: " + str(game_mode)
        print "Should render: " + str(render)
        print "Total step limit: " + str(total_step_limit)
        return game_name, game_mode, render, total_step_limit

    def _game_model(self, game_mode,game_name, action_space):
        if game_mode == "ddqn_training":
            return DDQNTrainer(game_name, INPUT_SHAPE, action_space)
        elif game_mode == "ddqn_testing":
            return DDQNSolver(game_name, INPUT_SHAPE, action_space)
        elif game_mode == "ge_training":
            return GETrainer(game_name, INPUT_SHAPE, action_space)
        elif game_mode == "ge_testing":
            return GESolver(game_name, INPUT_SHAPE, action_space)
        else:
            print "Unrecognized mode. Use --help"
            exit(1)


if __name__ == "__main__":
    Atari()

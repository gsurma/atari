import gym
import argparse
import numpy as np
import atari_py
from game_models.ddqn_game_model import DDQNTrainer, DDQNSolver
from game_models.ge_game_model import GETrainer, GESolver
from gym_wrappers import MainGymWrapper

FRAMES_IN_OBSERVATION = 4
FRAME_SIZE = 84
INPUT_SHAPE = (FRAMES_IN_OBSERVATION, FRAME_SIZE, FRAME_SIZE)


class Atari:

    def __init__(self):
        game_name, game_mode, render, total_step_limit, total_run_limit, clip = self._args()
        env_name = game_name + "Deterministic-v4"  # Handles frame skipping (4) at every iteration
        env = MainGymWrapper.wrap(gym.make(env_name))
        self._main_loop(self._game_model(game_mode, game_name, env.action_space.n), env, render, total_step_limit, total_run_limit, clip)

    def _main_loop(self, game_model, env, render, total_step_limit, total_run_limit, clip):
        if isinstance(game_model, GETrainer):
            game_model.genetic_evolution(env)

        run = 0
        total_step = 0
        while True:
            if total_run_limit is not None and run >= total_run_limit:
                print "Reached total run limit of: " + str(total_run_limit)
                exit(0)

            run += 1
            current_state = env.reset()
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
                next_state, reward, terminal, info = env.step(action)
                if clip:
                    np.sign(reward)
                score += reward
                game_model.remember(current_state, action, reward, next_state, terminal)
                current_state = next_state

                game_model.step_update(total_step)

                if terminal:
                    game_model.save_run(score, step, run)
                    break

    def _args(self):
        parser = argparse.ArgumentParser()
        available_games = list((''.join(x.capitalize() or '_' for x in word.split('_')) for word in atari_py.list_games()))
        parser.add_argument("-g", "--game", help="Choose from available games: " + str(available_games) + ". Default is 'breakout'.", default="Breakout")
        parser.add_argument("-m", "--mode", help="Choose from available modes: ddqn_train, ddqn_test, ge_train, ge_test. Default is 'ddqn_training'.", default="ddqn_training")
        parser.add_argument("-r", "--render", help="Choose if the game should be rendered. Default is 'False'.", default=False, type=bool)
        parser.add_argument("-tsl", "--total_step_limit", help="Choose how many total steps (frames visible by agent) should be performed. Default is '5000000'.", default=5000000, type=int)
        parser.add_argument("-trl", "--total_run_limit", help="Choose after how many runs we should stop. Default is None (no limit).", default=None, type=int)
        parser.add_argument("-c", "--clip", help="Choose whether we should clip rewards to (0, 1) range. Default is 'True'", default=True, type=bool)
        args = parser.parse_args()
        game_mode = args.mode
        game_name = args.game
        render = args.render
        total_step_limit = args.total_step_limit
        total_run_limit = args.total_run_limit
        clip = args.clip
        print "Selected game: " + str(game_name)
        print "Selected mode: " + str(game_mode)
        print "Should render: " + str(render)
        print "Should clip: " + str(clip)
        print "Total step limit: " + str(total_step_limit)
        print "Total run limit: " + str(total_run_limit)
        return game_name, game_mode, render, total_step_limit, total_run_limit, clip

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

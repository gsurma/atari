import gym
import numpy as np
import argparse
from PIL import Image
from game_models.ddqn_game_model import DDQNTrainer, DDQNSolver
from game_models.ge_game_model import GETrainer, GESolver

FRAMES_IN_OBSERVATION = 4
FRAME_SIZE = 84
TOTAL_STEP_LIMIT = 10000000


class Atari:

    def __init__(self):
        game_name, game_mode, render = self._args()
        env_name = game_name + "Deterministic-v4"  # Handles frame skipping (4) at every iteration
        env = gym.make(env_name)
        self._main_loop(self._game_model(game_mode, game_name, env.action_space.n), env, render)

    def _main_loop(self, game_model, env, render):
        run = 0
        total_step = 0
        while TOTAL_STEP_LIMIT:
            run += 1
            initial_state = env.reset()
            observation = self._preprocess_observation(initial_state)
            current_state = np.array([observation]*FRAMES_IN_OBSERVATION)

            step = 0
            score = 0
            while True:
                total_step += 1
                step += 1

                if render:
                    env.render()

                action = game_model.move(current_state)
                state, reward, terminal, info = env.step(action)
                reward = np.clip(reward, -1, 1)
                score += reward

                next_state = np.append(current_state[1:], [self._preprocess_observation(state)], axis=0)
                game_model.remember(current_state, action, reward, next_state, terminal)
                current_state = next_state

                game_model.step_update(total_step)

                if terminal:
                    game_model.save_run(score, step, run)
                    break

    def _preprocess_observation(self, obs):
        image = Image.fromarray(obs, "RGB").convert("L").resize((FRAME_SIZE, FRAME_SIZE))
        return np.asarray(image.getdata(), dtype=np.uint8).reshape(image.size[1], image.size[0])

    def _args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("-g", "--game", help="Choose from available games: Breakout, Pong, SpaceInvaders", default="Breakout") #TODO list
        parser.add_argument("-m", "--mode", help="Choose from available modes: ddqn_train, ddqn_test, ge_train, ge_test", default="ddqn_train")
        parser.add_argument("-r", "--render", help="Choose if the game should be rendered.", default=False)
        args = parser.parse_args()
        game_mode = args.mode
        game_name = args.game
        render = args.render
        print "Selected game: " + str(game_name)
        print "Selected mode: " + str(game_mode)
        print "Should render: " + str(render)
        return game_name, game_mode, render

    def _game_model(self, game_mode,game_name, action_space):
        if game_mode == "ddqn_train":
            return DDQNTrainer(game_name, FRAME_SIZE, action_space)
        elif game_mode == "ddqn_test":
            return DDQNSolver(game_name, FRAME_SIZE, action_space)
        elif game_mode == "ge_train":
            return GETrainer(game_name, FRAME_SIZE, action_space)
        elif game_mode == "ge_test":
            return GESolver(game_name, FRAME_SIZE, action_space)
        else:
            print "Unrecognized mode. Use --help"
            exit(1)


if __name__ == "__main__":
    Atari()

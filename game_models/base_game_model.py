
from scores.score_logger import ScoreLogger


class BaseGameModel:

    def __init__(self, game_name, mode_name, path, observation_space, action_space):
        self.action_space = action_space
        self.observation_space = observation_space
        self.score_logger = ScoreLogger(game_name + " " + mode_name, path)

    def save_run(self, score, steps):
        self.score_logger.add_score(score)
        self.score_logger.add_run_duration(steps)

    def save_model(self):
        pass

    def get_move(self, state):
        pass

    def move(self, state):
        pass

    def remember(self, state, action, reward, next_state, done):
        pass

    def step_update(self, step):
        pass


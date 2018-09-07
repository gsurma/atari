
from scores.score_logger import ScoreLogger


class BaseGameModel:

    def __init__(self, env_name, mode_name, csv_path, png_path, observation_space, action_space):
        self.action_space = action_space
        self.observation_space = observation_space
        self.score_logger = ScoreLogger(env_name + " " + mode_name,
                                        csv_path,
                                        png_path)

    def save_score(self, score):
        self.score_logger.add_score(score)

    def get_move(self, state):
        pass

    def move(self, state):
        pass

    def remember(self, state, action, reward, next_state, done):
        pass

    def experience_replay(self):
        pass

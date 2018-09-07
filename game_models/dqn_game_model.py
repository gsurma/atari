
from game_models.base_game_model import BaseGameModel


class DQNGameModel(BaseGameModel):

    def __init__(self, env_name, observation_space, action_space):
        BaseGameModel.__init__(self, env_name, "DQN", "./scores/dqn/scores.csv", "./scores/dqn/scores.png", observation_space, action_space)


class DQNSolver(DQNGameModel):

    def __init__(self, env_name, observation_space, action_space):
        DQNGameModel.__init__(self, env_name, observation_space, action_space)

    def move(self, state):
        pass
        #TODO


class DQNTrainer(DQNGameModel):

    def __init__(self, env_name, observation_space, action_space):
        DQNGameModel.__init__(self, env_name, observation_space, action_space)

    def move(self, state):
        pass
        #TODO

    def remember(self, state, action, reward, next_state, done):
        pass
        #TODO

    def experience_replay(self):
        pass
        #TODO
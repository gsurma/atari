
import random
from game_models.base_game_model import BaseGameModel


class DQNGameModel(BaseGameModel):

    def __init__(self, game_name, observation_space, action_space):
        BaseGameModel.__init__(self, game_name,
                               "DQN",
                               "./scores/" + game_name + "/dqn/scores.csv",
                               "./scores/" + game_name + "/dqn/scores.png",
                               observation_space,
                               action_space)


class DQNSolver(DQNGameModel):

    def __init__(self, game_name, observation_space, action_space):
        DQNGameModel.__init__(self, game_name, observation_space, action_space)

    def move(self, state):
        return random.choice(range(self.action_space))
        #TODO


class DQNTrainer(DQNGameModel):

    def __init__(self, game_name, observation_space, action_space):
        DQNGameModel.__init__(self, game_name, observation_space, action_space)

    def move(self, state):
        return random.choice(range(self.action_space))
        pass
        #TODO

    def remember(self, state, action, reward, next_state, done):
        pass
        #TODO

    def experience_replay(self):
        pass
        #TODO
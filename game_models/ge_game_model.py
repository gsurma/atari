import random
from game_models.base_game_model import BaseGameModel


class GEGameModel(BaseGameModel):

    def __init__(self, game_name, observation_space, action_space):
        BaseGameModel.__init__(self,
                               game_name,
                               "GE",
                               "./scores/" + game_name + "/ge/scores.csv",
                               "./scores/" + game_name + "/ge/scores.png",
                               observation_space,
                               action_space)


class GESolver(GEGameModel):

    def __init__(self, game_name, observation_space, action_space):
        GEGameModel.__init__(self, game_name, observation_space, action_space)

    def move(self, state):
        return random.choice(range(self.action_space))
        #TODO


class GETrainer(GEGameModel):

    def __init__(self, game_name, observation_space, action_space):
        GEGameModel.__init__(self, game_name, observation_space, action_space)

    def move(self, state):
        return random.choice(range(self.action_space))
        #TODO

    def remember(self, state, action, reward, next_state, done):
        pass
        #TODO

    def experience_replay(self):
        pass
        #TODO
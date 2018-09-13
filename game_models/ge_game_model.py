import random
from game_models.base_game_model import BaseGameModel


class GEGameModel(BaseGameModel):

    def __init__(self, game_name, mode_name, observation_space, action_space, path):
        BaseGameModel.__init__(self,
                               game_name,
                               mode_name,
                               path,
                               observation_space,
                               action_space)


class GESolver(GEGameModel):

    def __init__(self, game_name, observation_space, action_space):
        GEGameModel.__init__(self, game_name, "GE testing", observation_space, action_space, "./logs/" + game_name + "/ge/testing/")

    def move(self, state):
        return random.choice(range(self.action_space))
        #TODO


class GETrainer(GEGameModel):

    def __init__(self, game_name, observation_space, action_space):
        GEGameModel.__init__(self, game_name, "GE training", observation_space, action_space, "./logs/" + game_name + "/ge/training/")

    def move(self, state):
        return random.choice(range(self.action_space))
        #TODO

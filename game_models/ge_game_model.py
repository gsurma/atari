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
        GEGameModel.__init__(self, game_name, "GE test", observation_space, action_space, "./scores/" + game_name + "/ge/test/")

    def move(self, state):
        return random.choice(range(self.action_space))
        #TODO


class GETrainer(GEGameModel):

    def __init__(self, game_name, observation_space, action_space):
        GEGameModel.__init__(self, game_name, "GE train", observation_space, action_space, "./scores/" + game_name + "/ge/train/")

    def move(self, state):
        return random.choice(range(self.action_space))
        #TODO

    def save_model(self):
        pass

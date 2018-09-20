import random
from game_models.base_game_model import BaseGameModel


class GEGameModel(BaseGameModel):

    def __init__(self, game_name, mode_name, input_shape, action_space, logger_path):
        BaseGameModel.__init__(self,
                               game_name,
                               mode_name,
                               logger_path,
                               input_shape,
                               action_space)


class GESolver(GEGameModel):

    def __init__(self, game_name, input_shape, action_space):
        GEGameModel.__init__(self, game_name, "GE testing", input_shape, action_space, "./output/logs/" + game_name + "/ge/testing/" + self._get_date() + "/")

    def move(self, state):
        return random.choice(range(self.action_space))
        #TODO


class GETrainer(GEGameModel):

    def __init__(self, game_name, input_shape, action_space):
        GEGameModel.__init__(self, game_name, "GE training", input_shape, action_space, "./output/logs/" + game_name + "/ge/training/"+ self._get_date() + "/")

    def move(self, state):
        return random.choice(range(self.action_space))
        #TODO

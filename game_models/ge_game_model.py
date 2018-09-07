
from game_models.base_game_model import BaseGameModel


class GEGameModel(BaseGameModel):

    def __init__(self, env_name, observation_space, action_space):
        BaseGameModel.__init__(self, env_name, "GE", "./scores/ge/scores.csv", "./scores/ge/scores.png", observation_space, action_space)


class GESolver(GEGameModel):

    def __init__(self, env_name, observation_space, action_space):
        GEGameModel.__init__(self, env_name, observation_space, action_space)

    def move(self, state):
        pass
        #TODO


class GETrainer(GEGameModel):

    def __init__(self, env_name, observation_space, action_space):
        GEGameModel.__init__(self, env_name, observation_space, action_space)

    def move(self, state):
        pass
        #TODO

    def remember(self, state, action, reward, next_state, done):
        pass
        #TODO

    def experience_replay(self):
        pass
        #TODO
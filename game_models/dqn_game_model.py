from collections import deque
import random
import keras
import numpy as np
from game_models.base_game_model import BaseGameModel

GAMMA = 0.95
LEARNING_RATE = 0.001

MEMORY_SIZE = 1000000
BATCH_SIZE = 20

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.995


class DQNGameModel(BaseGameModel):

    def __init__(self, game_name, observation_space, action_space):
        BaseGameModel.__init__(self, game_name,
                               "DQN",
                               "./scores/" + game_name + "/dqn/scores.csv",
                               "./scores/" + game_name + "/dqn/scores.png",
                               observation_space,
                               action_space)

        self.exploration_rate = EXPLORATION_MAX

        self.action_space = action_space
        self.memory = deque(maxlen=MEMORY_SIZE)

        frames_input = keras.layers.Input((observation_space, observation_space, 1), name='frames')
        actions_input = keras.layers.Input((self.action_space,), name='mask')
        normalized = keras.layers.Lambda(lambda x: x / 255.0)(frames_input)
        conv_1 = keras.layers.convolutional.Convolution2D(16, 8, 8, subsample=(4, 4), activation='relu')(normalized)
        conv_2 = keras.layers.convolutional.Convolution2D(32, 4, 4, subsample=(2, 2), activation='relu')(conv_1)
        conv_flattened = keras.layers.core.Flatten()(conv_2)
        hidden = keras.layers.Dense(256, activation='relu')(conv_flattened)
        output = keras.layers.Dense(self.action_space)(hidden)
        filtered_output = keras.layers.multiply([output, actions_input])#keras.layers.merge([output, actions_input], mode='mul')

        self.model = keras.models.Model(input=[frames_input, actions_input], output=filtered_output)
        optimizer = keras.optimizers.RMSprop(lr=0.00025, rho=0.95, epsilon=0.01)
        self.model.compile(optimizer, loss='mse')
        # TODO: loading/saving weights


class DQNSolver(DQNGameModel):

    def __init__(self, game_name, observation_space, action_space):
        DQNGameModel.__init__(self, game_name, observation_space, action_space)

    def move(self, state):
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])


class DQNTrainer(DQNGameModel):

    def __init__(self, game_name, observation_space, action_space):
        DQNGameModel.__init__(self, game_name, observation_space, action_space)

    def move(self, state):
        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.action_space)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])
        #return random.choice(range(self.action_space))
        pass
        #TODO

    def remember(self, state, action, reward, next_state, done):
        pass
        #TODO

    def experience_replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE)
        for state, action, reward, state_next, terminal in batch:
            q_update = reward
            if not terminal:
                q_update = (reward + GAMMA * np.amax(self.model.predict(state_next)[0]))
            q_values = self.model.predict(state)
            q_values[0][action] = q_update
            self.model.fit(state, q_values, verbose=0)
        self.exploration_rate *= EXPLORATION_DECAY
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)
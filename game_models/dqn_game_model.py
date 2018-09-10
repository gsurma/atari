from collections import deque
import random
from statistics import mean
from keras.optimizers import RMSprop
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense
import numpy as np
import os
from game_models.base_game_model import BaseGameModel

GAMMA = 0.99

LEARNING_RATE = 0.00025
GRADIENT_MOMENTUM = 0.95
MIN_SQUARED_GRADIENT = 0.01

MEMORY_SIZE = 1000000
BATCH_SIZE = 32

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.1
EXPLORATION_STEPS = 500000#1000000
EXPLORATION_DECAY = (EXPLORATION_MAX-EXPLORATION_MIN)/EXPLORATION_STEPS


class DQNGameModel(BaseGameModel):

    def __init__(self, game_name, observation_space, action_space):
        BaseGameModel.__init__(self, game_name,
                               "DQN",
                               "./scores/" + game_name + "/dqn/",
                               observation_space,
                               action_space)

        self.exploration_rate = EXPLORATION_MAX
        self.model_path = "./tf_models/" + game_name + "/dqn/model.h5"
        if not os.path.exists(os.path.dirname(self.model_path)):
            os.makedirs(os.path.dirname(self.model_path))
        self.action_space = action_space
        self.memory = deque(maxlen=MEMORY_SIZE)

        input_shape = (observation_space, observation_space, 4)
        self.model = Sequential()
        self.model.add(Conv2D(32, 8,
                              strides=(4, 4),
                              activation='relu',
                              input_shape=input_shape))
        self.model.add(Conv2D(64, 4,
                              strides=(2, 2),
                              activation='relu',
                              input_shape=input_shape))
        self.model.add(Conv2D(64, 3,
                              strides=(1, 1),
                              activation='relu',
                              input_shape=input_shape))
        self.model.add(Flatten())
        self.model.add(Dense(512, activation='relu'))
        self.model.add(Dense(self.action_space))

        optimizer = RMSprop(lr=LEARNING_RATE, rho=GRADIENT_MOMENTUM, epsilon=MIN_SQUARED_GRADIENT)
        self.model.compile(loss='mse',
                           optimizer=optimizer)
        if os.path.isfile(self.model_path):
            self.model.load_weights(self.model_path)


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

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def experience_replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE)
        losses = []
        for state, action, reward, state_next, terminal in batch:
            q_update = reward
            if not terminal:
                prediction = self.model.predict(state_next)[0]
                print "prediction next: " + str(prediction)
                predictions = self.model.predict(state)[0]
                print "prediction state: " + str(predictions)
                #exit()
                q_update = reward + GAMMA * np.amax(prediction)
                print "q update: " + str(q_update)

            q_values = self.model.predict(state)
            q_values[0][action] = q_update
            print "q: " + str(q_values)
            fit = self.model.fit(state, q_values, verbose=0)
            loss = fit.history["loss"][0]
            losses.append(loss)
        return mean(losses)

    def update_exploration_rate(self):
        self.exploration_rate -= EXPLORATION_DECAY
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)

    def save_model(self):
        self.model.save_weights(self.model_path)
from collections import deque
from statistics import mean
from keras.optimizers import RMSprop
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense
import numpy as np
import os
import random
import cPickle as pickle
from game_models.base_game_model import BaseGameModel

GAMMA = 0.99

LEARNING_RATE = 0.001

MEMORY_SIZE = 1000000
BATCH_SIZE = 32

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.1
EXPLORATION_TEST = 0.05
EXPLORATION_STEPS = 1000000
EXPLORATION_DECAY = (EXPLORATION_MAX-EXPLORATION_MIN)/EXPLORATION_STEPS

UPDATE_FREQUENCY = 4
TARGET_NETWORK_UPDATE_FREQUENCY = 10000
LEARNING_STATS_UPDATE_FREQUENCY = 1000
DATA_PERSISTENCE_UPDATE_FREQUENCY = 10000
REPLAY_START_SIZE = 50000


class DQNModel:

    def __init__(self, input_shape, action_space):
        self.model = Sequential()
        self.model.add(Conv2D(32,
                              8,
                              strides=(4, 4),
                              padding="valid",
                              activation="relu",
                              input_shape=input_shape,
                              data_format="channels_first"))
        self.model.add(Conv2D(64,
                              4,
                              strides=(2, 2),
                              padding="valid",
                              activation="relu",
                              input_shape=input_shape,
                              data_format="channels_first"))
        self.model.add(Conv2D(64,
                              3,
                              strides=(1, 1),
                              padding="valid",
                              activation="relu",
                              input_shape=input_shape,
                              data_format="channels_first"))
        self.model.add(Flatten())
        self.model.add(Dense(512, activation="relu"))
        self.model.add(Dense(action_space))
        self.model.compile(loss="mean_squared_error",
                           optimizer=RMSprop(lr=LEARNING_RATE),
                           metrics=["accuracy"])
        self.model.summary()


class DQNGameModel(BaseGameModel):

    def __init__(self, game_name, mode_name, observation_space, action_space, path):
        BaseGameModel.__init__(self, game_name,
                               mode_name,
                               path,
                               observation_space,
                               action_space)

        self.input_shape = (4, observation_space, observation_space)

        self.model_path = "./neural_nets/" + game_name + "/dqn/model.h5"
        if not os.path.exists(os.path.dirname(self.model_path)):
            os.makedirs(os.path.dirname(self.model_path))

        self.memory_path = "./memory_dumps/" + game_name + "/dqn/memory.pickle"
        if not os.path.exists(os.path.dirname(self.memory_path)):
            os.makedirs(os.path.dirname(self.memory_path))

        self.action_space = action_space

        self.dqn = DQNModel(self.input_shape, action_space).model
        self._load_model()

        self.memory = deque(maxlen=MEMORY_SIZE)
        self._load_memory()

    def _save_model(self):
        self.dqn.save_weights(self.model_path)

    def _load_model(self):
        if os.path.isfile(self.model_path):
            self.dqn.load_weights(self.model_path)

    def _save_memory(self):
        with open(self.memory_path, "wb") as handle:
            pickle.dump(self.memory, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def _load_memory(self):
        if os.path.isfile(self.memory_path):
            with open(self.memory_path, "rb") as handle:
                self.memory = pickle.load(handle)


class DQNSolver(DQNGameModel):

    def __init__(self, game_name, observation_space, action_space):
        DQNGameModel.__init__(self, game_name, "DQN testing", observation_space, action_space, "./logs/" + game_name + "/dqn/testing/")

    def move(self, state):
        if np.random.rand() < EXPLORATION_TEST:
            return random.randrange(self.action_space)
        q_values = self.dqn.predict(np.asarray([state]).astype(np.float64), batch_size=1)
        return np.argmax(q_values[0])


class DQNTrainer(DQNGameModel):

    losses = []
    accuracies = []
    #TODO: q values

    def __init__(self, game_name, observation_space, action_space):
        DQNGameModel.__init__(self, game_name, "DQN training", observation_space, action_space, "./logs/" + game_name + "/dqn/training/")
        self.dqn_target = DQNModel(self.input_shape, action_space).model
        self._reset_target_network()
        self.epsilon = EXPLORATION_MAX
        self.training_count = 0

    def move(self, state):
        if np.random.rand() < self.epsilon or len(self.memory) < REPLAY_START_SIZE:
            return random.randrange(self.action_space)
        q_values = self.dqn.predict(np.asarray([state]).astype(np.float64), batch_size=1)
        return np.argmax(q_values[0])

    def remember(self, current_state, action, reward, next_state, terminal):
        self.memory.append({"current_state": np.asarray([current_state]),
                            "action": action,
                            "reward": reward,
                            "next_state": np.asarray([next_state]),
                            "terminal": terminal})

    def step_update(self, total_step):
        if total_step % UPDATE_FREQUENCY == 0 and len(self.memory) >= REPLAY_START_SIZE:
            loss, accuracy = self._train()
            self.losses.append(loss)
            self.accuracies.append(accuracy)

        if self.training_count % LEARNING_STATS_UPDATE_FREQUENCY == 0 and self.losses and self.accuracies:
            self.logger.add_accuracy(mean(self.accuracies))
            self.accuracies = []
            self.logger.add_loss(mean(self.losses))
            self.losses = []

        if self.training_count % TARGET_NETWORK_UPDATE_FREQUENCY == 0 and self.training_count >= TARGET_NETWORK_UPDATE_FREQUENCY:
            self._reset_target_network()

        if len(self.memory) >= REPLAY_START_SIZE:
            self._update_epsilon()

        if self.training_count % DATA_PERSISTENCE_UPDATE_FREQUENCY == 0 and self.training_count >= DATA_PERSISTENCE_UPDATE_FREQUENCY:
            self._save_model()
            self._save_memory()

    def _train(self):
        batch = np.asarray(random.sample(self.memory, BATCH_SIZE))
        if len(batch) < BATCH_SIZE:
            return

        self.training_count += 1
        print "Training session: %d,  epsilon: %f" % (self.training_count, self.epsilon)

        current_states = []
        q_values = []

        for entry in batch:
            current_states.append(entry["current_state"].astype(np.float64))
            next_state = entry["next_state"].astype(np.float64)
            next_state_prediction = self.dqn_target.predict(next_state).ravel()
            next_q_value = np.max(next_state_prediction)
            q = list(self.dqn.predict(entry["current_state"])[0])
            if entry["terminal"]:
                q[entry["action"]] = entry["reward"]
            else:
                q[entry["action"]] = entry["reward"] + GAMMA * next_q_value
            q_values.append(q)

        fit = self.dqn.fit(np.asarray(current_states).squeeze(),
                           np.asarray(q_values).squeeze(),
                           batch_size=BATCH_SIZE)
        return fit.history["loss"][0], fit.history["acc"][0]

    def _update_epsilon(self):
        self.epsilon -= EXPLORATION_DECAY
        self.epsilon = max(EXPLORATION_MIN, self.epsilon)

    def _reset_target_network(self):
        self.dqn_target.set_weights(self.dqn.get_weights())





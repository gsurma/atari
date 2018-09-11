from collections import deque
import random
from statistics import mean
from keras.optimizers import RMSprop
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense
import numpy as np
import os
import random
from game_models.base_game_model import BaseGameModel

GAMMA = 0.99

LEARNING_RATE = 0.00025
# GRADIENT_MOMENTUM = 0.95
# MIN_SQUARED_GRADIENT = 0.01

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
REPLAY_START_SIZE = 50000


class DQNModel:

    def __init__(self, input_shape, action_space):
        self.model = Sequential()
        self.model.add(Conv2D(32, 8, strides=(4, 4),
                                padding='valid',
                                activation='relu',
                                input_shape=input_shape,
                                data_format='channels_first'))
        self.model.add(Conv2D(64, 4, strides=(2, 2),
                              padding='valid',
                              activation='relu',
                              input_shape=input_shape,
                              data_format='channels_first'))
        self.model.add(Conv2D(64, 3, strides=(1, 1),
                              padding='valid',
                              activation='relu',
                              input_shape=input_shape,
                              data_format='channels_first'))
        self.model.add(Flatten())
        self.model.add(Dense(512, activation='relu'))
        self.model.add(Dense(action_space))
        # optimizer = optimizer = keras.optimizers.RMSprop(
        #     lr=0.00025, rho=0.95, epsilon=0.01
        # )

        self.model.compile(loss='mean_squared_error',
                      optimizer=RMSprop(LEARNING_RATE),
                      metrics=['accuracy'])


class DQNGameModel(BaseGameModel):

    def __init__(self, game_name, mode_name, observation_space, action_space, path):
        BaseGameModel.__init__(self, game_name,
                               mode_name,
                               path,
                               observation_space,
                               action_space)

        self.input_shape = (4, observation_space, observation_space)

        self.model_path = "./tf_models/" + game_name + "/dqn/model.h5"
        if not os.path.exists(os.path.dirname(self.model_path)):
            os.makedirs(os.path.dirname(self.model_path))
        self.action_space = action_space

        self.dqn = DQNModel(self.input_shape, action_space).model
        if os.path.isfile(self.model_path):
            self.dqn.load_weights(self.model_path)


class DQNSolver(DQNGameModel):

    def __init__(self, game_name, observation_space, action_space):
        DQNGameModel.__init__(self, game_name,"DQN test", observation_space, action_space, "./scores/" + game_name + "/dqn/test/")

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
        DQNGameModel.__init__(self, game_name, "DQN train", observation_space, action_space, "./scores/" + game_name + "/dqn/train/")
        self.dqn_target = DQNModel(self.input_shape, action_space).model
        self.dqn_agent = DQNAgent(self.dqn, self.dqn_target, action_space, self.input_shape)
        self.dqn_agent.reset_target_network()

    def move(self, state):
        if np.random.rand() < self.dqn_agent.epsilon or len(self.dqn_agent.experiences) < REPLAY_START_SIZE:
            return random.randrange(self.action_space)
        q_values = self.dqn.predict(np.asarray([state]).astype(np.float64), batch_size=1)
        return np.argmax(q_values[0])

    def remember(self, state, action, reward, next_state, done):
        self.dqn_agent.add_experience(np.asarray([state]), action, reward, np.asarray([next_state]), done)
        # TODO:
        #self.memory.append((state, action, reward, next_state, done))

    def step_update(self, total_step):
        # Train the agent
        if total_step % UPDATE_FREQUENCY == 0 and len(self.dqn_agent.experiences) >= REPLAY_START_SIZE:
            loss, accuracy = self.dqn_agent.train()
            self.losses.append(loss)
            self.accuracies.append(accuracy)

        if self.dqn_agent.training_count % LEARNING_STATS_UPDATE_FREQUENCY == 0 and self.losses and self.accuracies:
            self.score_logger.add_accuracy(mean(self.accuracies))
            self.accuracies = []
            self.score_logger.add_loss(mean(self.losses))
            self.losses = []

        # Every C DQN updates, update DQN_target
        if self.dqn_agent.training_count % TARGET_NETWORK_UPDATE_FREQUENCY == 0 and self.dqn_agent.training_count >= TARGET_NETWORK_UPDATE_FREQUENCY:
            self.dqn_agent.reset_target_network()
            # # Log the mean score and mean Q values of test states
            # if DQA.training_count % args.avg_val_computation_freq == 0 and DQA.training_count >= args.avg_val_computation_freq:
            #     logger.to_csv(test_csv,
            #                   [np.mean(test_scores), np.mean(test_mean_q)])
            #     del test_scores[:]
            #     del test_mean_q[:]

        # Linear epsilon annealing
        if len(self.dqn_agent.experiences) >= REPLAY_START_SIZE:
            self.dqn_agent.update_epsilon()

    def save_model(self):
        self.dqn.save_weights(self.model_path)


class DQNAgent:
    def __init__(self,
                 dqn,
                 dqn_target,
                 actions,
                 network_input_shape):

        self.dqn = dqn
        self.dqn_target = dqn_target
        self.network_input_shape = network_input_shape
        self.actions = actions
        self.epsilon = EXPLORATION_MAX
        self.experiences = deque(maxlen=MEMORY_SIZE)
        self.training_count = 0

    def add_experience(self, source, action, reward, dest, final):
        self.experiences.append({'source': source,
                                 'action': action,
                                 'reward': reward,
                                 'dest': dest,
                                 'final': final})

    def sample_batch(self):
        batch = []
        for i in xrange(BATCH_SIZE):
            batch.append(self.experiences[random.randrange(0, len(self.experiences))])
        return np.asarray(batch)

    def train(self):
        self.training_count += 1
        print 'Training session #%d - epsilon: %f' % \
              (self.training_count, self.epsilon)
        batch = self.sample_batch()

        x_train = []
        t_train = []

        # Generate training inputs and targets
        for datapoint in batch:
            # Inputs are the states
            x_train.append(datapoint['source'].astype(np.float64))

            # Apply the DQN or DDQN Q-value selection
            next_state = datapoint['dest'].astype(np.float64)
            next_state_pred = self.dqn_target.predict(next_state).ravel()
            next_q_value = np.max(next_state_pred)

            # The error must be 0 on all actions except the one taken
            t = list(self.dqn.predict(datapoint['source'])[0])
            if datapoint['final']:
                t[datapoint['action']] = datapoint['reward']
            else:
                t[datapoint['action']] = datapoint['reward'] + GAMMA * next_q_value
            t_train.append(t)

        # Train the model for one epoch
        h = self.dqn.fit(np.asarray(x_train).squeeze(),
                         np.asarray(t_train).squeeze(),
                         batch_size=BATCH_SIZE)

        return h.history['loss'][0], h.history['acc'][0]

    def update_epsilon(self):
        self.epsilon -= EXPLORATION_DECAY
        self.epsilon = max(EXPLORATION_MIN, self.epsilon)

    def reset_target_network(self):
        print 'Updating target network...'
        self.dqn_target.set_weights(self.dqn.get_weights())


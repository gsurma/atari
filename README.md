<h3 align="center">
  <img src="assets/atari_icon_web.png" width="300">
</h3>

# Atari

Research Playground built on top of [OpenAI's Atari Gym](https://gym.openai.com/envs/#atari), prepared for implementing various Reinforcement Learning algorithms.

It can emulate any of the following games:

> ['Asterix', 'Asteroids',
>                         'MsPacman', 'Kaboom', 'BankHeist', 'Kangaroo',
>                         'Skiing', 'FishingDerby', 'Krull', 'Berzerk',
>                         'Tutankham', 'Zaxxon', 'Venture', 'Riverraid',
>                         'Centipede', 'Adventure', 'BeamRider', 'CrazyClimber',
>                         'TimePilot', 'Carnival', 'Tennis', 'Seaquest',
>                         'Bowling', 'SpaceInvaders', 'Freeway', 'YarsRevenge',
>                         'RoadRunner', 'JourneyEscape', 'WizardOfWor',
>                         'Gopher', 'Breakout', 'StarGunner', 'Atlantis',
>                         'DoubleDunk', 'Hero', 'BattleZone', 'Solaris',
>                         'UpNDown', 'Frostbite', 'KungFuMaster', 'Pooyan',
>                         'Pitfall', 'MontezumaRevenge', 'PrivateEye',
>                         'AirRaid', 'Amidar', 'Robotank', 'DemonAttack',
>                         'Defender', 'NameThisGame', 'Phoenix', 'Gravitar',
>                         'ElevatorAction', 'Pong', 'VideoPinball', 'IceHockey',
>                         'Boxing', 'Assault', 'Alien', 'Qbert', 'Enduro',
>                         'ChopperCommand', 'Jamesbond']

## Purpose
Ultimate goal of this project is to implement and compare various RL approaches with atari games as a common denominator.

## Usage

1. Clone the repo.
2. Go to the project's root folder.
3. Install required packages`pip install -r requirements.txt`.
4. Launch atari. I recommend starting with help command to see all available modes `python atari.py --help`.


## Modes

All below modes were benchmarked using following


10M - 85h on Tesla K80
5M ~40h on Tesla K80

Breakout (human: 28.3)
- [ ] DDQN
- [ ] GE

SpaceInvaders (human: 372.5)
- [ ] DDQN
- [ ] GE




GAMMA = 0.99
MEMORY_SIZE = 900000
BATCH_SIZE = 32
TRAINING_FREQUENCY = 4
TARGET_NETWORK_UPDATE_FREQUENCY = TRAINING_FREQUENCY*10000
MODEL_PERSISTENCE_UPDATE_FREQUENCY = 10000
REPLAY_START_SIZE = 50000

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.1
EXPLORATION_TEST = 0.01
EXPLORATION_STEPS = 850000
EXPLORATION_DECAY = (EXPLORATION_MAX-EXPLORATION_MIN)/EXPLORATION_STEPS


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
                           optimizer=RMSprop(lr=0.00025,
                                             rho=0.95,
                                             epsilon=0.01),
                           metrics=["accuracy"])
        self.model.summary()
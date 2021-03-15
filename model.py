import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.losses import Huber

import config


def goose_model(n_actions):
    frames_input = Input(config.atari_shape, name='frames')
    actions_input = Input((n_actions,), name='mask')

    normalized = Lambda(lambda x: x / 255.0)(frames_input)

    conv_1 = Conv2D(32, kernel_size=(8, 8), strides=(4, 4), activation='relu', data_format='channels_first')(normalized)
    conv_2 = Conv2D(64, kernel_size=(4, 4), strides=(2, 2), activation='relu', data_format='channels_first')(conv_1)
    conv_3 = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='relu', data_format='channels_first')(conv_2)

    conv_flattened = Flatten()(conv_3)
    hidden = Dense(512, activation='relu')(conv_flattened)
    output = Dense(n_actions)(hidden)

    filtered_output = multiply([output, actions_input])

    model = Model(inputs=[frames_input, actions_input], outputs=filtered_output)
    optimizer = RMSprop(lr=config.learning_rate, rho=0.95, epsilon=0.01)
    model.compile(optimizer, loss=Huber)

    return model


def fit_batch(model, target_model, start_states, actions, rewards, next_states, is_terminals):
    """
    Do one deep Q learning iteration.
    Params:
    - model: The DQN
    - start_states: numpy array of starting states
    - actions: numpy array of one-hot encoded actions corresponding to the start states
    - rewards: numpy array of rewards corresponding to the start states and actions
    - next_states: numpy array of the resulting states corresponding to the start states and actions
    - is_terminals: numpy boolean array of whether the resulting state is terminal
    """

    # First, predict the Q values of the next states. Note how we are passing ones as the mask.
    next_q_values = target_model.predict([next_states, np.ones((config.batch_size, config.num_actions))])

    # The Q values of the terminal states is 0 by definition, so override them
    next_q_values[is_terminals] = 0

    # The Q values of each start state is the reward + gamma * the max next state Q value
    q_values = rewards + config.gamma * np.max(next_q_values, axis=1)

    # Fit the keras model.
    # Note how we are passing the actions as the mask and multiplying the targets by the actions.
    actions_target = np.eye(config.num_actions)[np.array(actions).reshape(-1)]
    targets = actions_target * q_values[:, None]
    model.fit(x=[start_states, actions_target], y=targets, batch_size=config.batch_size, epochs=1, verbose=0)


def choose_best_action(model, state):
    state_reshape = np.reshape(state, (1, config.atari_shape[0], config.atari_shape[1], config.atari_shape[2]))
    q_value = model.predict([state_reshape, np.ones((1, config.num_actions))], batch_size=1)
    return np.argmax(q_value[0])

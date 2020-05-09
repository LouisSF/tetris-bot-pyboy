#!/usr/bin/python3

# print(Q_values)
# print(Q_values.shape)
# print(Q_values[0])
# print(np.argmax(Q_values[0]))

import tensorflow as tf
import tensorflow.random
import keras
import numpy as np

import gym
from collections import deque

import os
import sys

from pyboy import PyBoy
from pyboy import WindowEvent
from pyboy.plugins import game_wrapper_tetris

'''
This time we consider not immediate inputs to provide,
but rather final states
'''

'''
Important : CHECK ALL VALID STATES BEFORE SENDING INPUTS TO EMULATOR
'''

np.random.seed(42)
tf.random.set_random_seed(42)

def epsilon_greedy_policy(state, epsilon=0):
    # epsilon = 0.01
    if np.random.rand() < epsilon:
        return np.random.randint(5)
    else:
        Q_values = model.predict(state[np.newaxis])

        return np.argmax(Q_values[0])


def training_step(batch_size):
    experiences = sample_experiences(batch_size)
    numpy_game_areas, actions, rewards, new_numpy_game_areas, dones = experiences
    next_Q_values = model.predict(new_numpy_game_areas)
    print("NEXT Q VALUES ", next_Q_values)
    max_next_Q_values = np.max(next_Q_values, axis=2)
    print("MAX NEXT Q VALUES ",max_next_Q_values)
    print("SHAPES")
    print("REWARDS : ", rewards.shape)
    print("DONES", dones.shape)
    print("NEXTQVALUES", next_Q_values.shape)
    print("MAXNEXTQVALUES", max_next_Q_values.shape)
    target_Q_values = (rewards + (1-dones) * discount_factor * max_next_Q_values)
    mask = tf.one_hot(actions, n_outputs)
    with tf.GradientTape() as tape:
        all_Q_values = model(numpy_game_areas)
        Q_values = tf.reduce_sum(all_Q_values * mask, axis = 1, keepdims=True)
        loss = tf.reduce_mean(loss_fn(target_Q_values, Q_values))
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

def sample_experiences(batch_size):
    indices = np.random.randint(len(replay_buffer), size=batch_size)
    batch = [replay_buffer[index] for index in indices]
    print("Length of batch", len(batch))
    # print(batch)
    numpy_game_areas, actions, rewards, new_numpy_game_areas, dones = [
        np.array([experience[field_index] for experience in batch])
        for field_index in range(5)]
    return numpy_game_areas, actions, rewards, new_numpy_game_areas, dones

#tetris, pyboy, new_game_area, epsilon
#env, state, epsilon
def play_one_frame(tetris, pyboy, numpy_game_area, epsilon):
    old_lines = tetris.lines
    action = epsilon_greedy_policy(numpy_game_area, epsilon)
    #new_game_area, reward, done = env.step(action)

    if action==0:
        pyboy.send_input(WindowEvent.PRESS_ARROW_LEFT)
        pyboy.tick()
        pyboy.send_input(WindowEvent.RELEASE_ARROW_LEFT)
    elif action==1:
        pyboy.send_input(WindowEvent.PRESS_ARROW_RIGHT)
        pyboy.tick()
        pyboy.send_input(WindowEvent.RELEASE_ARROW_RIGHT)
    elif action==2:
        pyboy.send_input(WindowEvent.PRESS_BUTTON_A)
        pyboy.tick()
        pyboy.send_input(WindowEvent.RELEASE_BUTTON_A)
    elif action==3:
        pyboy.send_input(WindowEvent.PRESS_BUTTON_B)
        pyboy.tick()
        pyboy.send_input(WindowEvent.RELEASE_BUTTON_B)
    elif action==4:
        pyboy.tick()
    else:
        print("ACTION", action)
        print("ERREUR, ACTION NON PREVUE")
    new_game_area = tetris.game_area()
    new_numpy_game_area = np.asarray(new_game_area)
    ''' WE DEFINE REWARD AS DIFFERENCE IN LINE CLEARED AFTER MOVED
    BUT IT CAN ALSO BE THE SCORE, THE DIFFERENCE IN SCORE BETWEEN 2 frames
    OR THE TOT NUMBER OF CLEARED LINES'''
    reward = tetris.lines - old_lines
    '''We are done (ie we lose) if we reach the top of the game area
    which means at least one column is full'''
    # print([all(filter(lambda x: x != blank_tile, new_game_area[:, i])) for i in range(new_numpy_game_area.shape[1])])
    # done = any([all(filter(lambda x: x != blank_tile, new_game_area[:, i])) for i in range(new_numpy_game_area.shape[1])])
    done = np.any([np.all(new_numpy_game_area[:,i] != np.full(new_numpy_game_area.shape, blank_tile)[:,i]) for i in range(new_numpy_game_area.shape[1])])
    replay_buffer.append((numpy_game_area, action, reward, new_numpy_game_area, done))
    return np.copy(new_game_area), reward, done

def main():

    for episode in range(10):
        print("EPISODE", episode)
        print("LENGTH OF REPLAY BUFFER", len(replay_buffer))
        first_brick = False
        #obs = env.reset()
        tetris.reset_game()
        old_game_area = tetris.game_area()
        old_numpy_game_area = np.asarray(old_game_area)
        new_numpy_game_area = np.copy(old_numpy_game_area)
        #for frame in range(4000): # Enough frames for the test. Otherwise do: `while not pyboy.tick():`
        done = False
        while not done:
            #print(tetris)
            #pyboy.tick()
            #print(frame)
            '''WE DONT WANT TO BE ANALYSING EVERY FRAME, ONLY EVERY
            FRAME WHERE NEW GAME AREA IS DIFFERENT FROM THE PREVIOUS ONE'''
            # Illustrating how we can extract the game board quite simply. This can be used to read the tile identifiers.
            # new_game_area = tetris.game_area()
            # new_numpy_game_area = np.asarray(new_game_area)
            if np.all(old_numpy_game_area==new_numpy_game_area):
                pyboy.tick()
                new_game_area = tetris.game_area()
                new_numpy_game_area = np.asarray(new_game_area)
            else:
                old_numpy_game_area = np.copy(new_numpy_game_area)
                epsilon = max(1-episode/500, 0.01)
                new_game_area, reward, done = play_one_frame(tetris, pyboy, new_numpy_game_area, epsilon)
                new_numpy_game_area = np.asarray(new_game_area)
                # if done:
                #     break

            # game_area is accessed as [<row>, <column>].
            # 'game_area[-1,:]' is asking for all (:) the columns in the last row (-1)
            if not first_brick and any(filter(lambda x: x != blank_tile, game_area[-1, :])):
                first_brick = True
                print("First brick touched the bottom!")
                print(tetris)
        if episode > -1:
            training_step(batch_size)

    print("Final game board mask:")
    print(tetris)

    # Assert there is something on the bottom of the game area
    #assert any(filter(lambda x: x != blank_tile, game_area[-1, :]))
    tetris.reset_game()
    # After reseting, we should have a clean game area
    #assert all(filter(lambda x: x != blank_tile, game_area[-1, :]))

    pyboy.stop()

n_outputs = 5 # == env.action_space.n # left(0), right(1), a(2), b(3), nothing(4)
batch_size = 32
discount_factor = 0.95
optimizer = keras.optimizers.Adam(lr=1e-3)
loss_fn = keras.losses.mean_squared_error
replay_buffer = deque(maxlen=10000)

pyboy = PyBoy('Tetris.GB',  game_wrapper=True, disable_renderer=False)
pyboy.set_emulation_speed(0)

assert pyboy.cartridge_title() == "TETRIS"

tetris = pyboy.game_wrapper()
tetris.start_game()

assert tetris.score == 0
assert tetris.level == 0
assert tetris.lines == 0
assert tetris.fitness == 0 # A built-in fitness score for AI development

blank_tile = 47

game_area = tetris.game_area()
numpy_game_area = np.asarray(game_area)
input_shape = (numpy_game_area.shape)
print(input_shape)

model = keras.models.Sequential([
    keras.layers.Dense(512, activation="elu", input_shape = input_shape),
    keras.layers.Dense(256, activation="elu"),
    #keras.layers.Dense(128, activation="elu"),
    keras.layers.Dense(n_outputs)
])

main()

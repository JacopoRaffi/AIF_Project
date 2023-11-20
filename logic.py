# file for the logic of the game

from typing import Tuple, List
import numpy as np
import gym
import matplotlib.pyplot as plt
import minihack
import random

from utils import get_valid_moves, get_player_location, action_map, get_boulder_locationV

def choose_best_action(valid_moves: List[Tuple[int, int]], game_map: np.ndarray, player_position: Tuple[int, int]) -> int: 
    """
        choose the best action for the agent so to find a water block(as a result, also the river)
        :param valid_moves: all the moves the agent can perform from its position
        :return: the best action to perform
    """

    action = -1
    actions = []
    boulder_positions = get_boulder_locationV(game_map)
    not_same_coords = [move for move in valid_moves if move not in boulder_positions]

    if len(not_same_coords) > 0: #if there is a step that doesn't move a boulder
        #prioritize steps that would move the player on different raw than the boulders
        not_same_y_coords = [move for move in not_same_coords if all(move[0] != boulder[0] for boulder in boulder_positions)]
            
        if not len(not_same_y_coords) > 0:
            actions = [action_map(player_position, move)[0] for move in not_same_coords]
        else:
            actions = [action_map(player_position, move)[0] for move in not_same_y_coords]  
    
    else: #almost impossible to happen (agent completely cornered by boulders)
        actions = [action_map(player_position, move)[0] for move in valid_moves]

    action = random.choice(actions)
    return action

 #This function needs to be exported in the notebook (it was here just for testing)


"""def find_river_coordinates(game_env: gym.Env, game_map: np.ndarray) -> List[Tuple[int, int]]:
    
        moves the player until a water block is found
        a river is assumed to be a vertical straight line of water blocks

        :param game_env: the environment of the game
        :param game_map: the initial map of the game
        :return: the coordinates of the water blocks of the river
    
    # check if the river is in the initial map
    river_coordinates = np.where(game_map == ord("}"))
    if len(river_coordinates[0]) > 0:
            found = True
            return river_coordinates

    # if the river is not seen player will pefrom moves until it's found
    found = False
    player_location = get_player_location(game_map)
    while not found: 
        action = choose_best_action(get_valid_moves(game_map, player_location), game_map, player_location)
        obs_state, _, _, _ = game_env.step(action)
        #plt.imshow(obs_state['pixel'][100:250, 400:750])
        print(action)
        river_coordinates = np.where(obs_state["chars"] == ord("}"))

        if len(river_coordinates[0]) > 0: #if the river is found
            found = True

        game_map = obs_state["chars"] #update observable map so to take the next "best action"

    return river_coordinates
"""

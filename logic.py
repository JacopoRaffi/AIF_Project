# file for the logic of the game

from typing import Tuple, List
import numpy as np
import gym
import matplotlib.pyplot as plt
import minihack
import random

from utils import *

def avoid_the_obstacle(valid_moves: List[Tuple[int, int]], game_map: np.ndarray, player_position: Tuple[int, int], obstacle_position: Tuple[int, int]):
    """
        manage the player stuck by an obstacle (boulder or river)
        :param valid_moves: all the moves the agent can perform from its position
        :param game_map: the game map as a matrix
        :param player_position: the current position of the agent
        :param obstacle_position: the position of the obstacle
    """

    # get the direction the player is following
    direction = action_map(player_position, obstacle_position)
    new_player_position = player_position

    # check if the obstacle is a river, else is a boulder
    if game_map[obstacle_position] == ord("}"):

        # the player is going East
        if direction[1] == "E":
            # vai a N -> SE -> SE
            new_player_position = (new_player_position[0] - 1, new_player_position[1])
            new_player_position = (new_player_position[0] + 1, new_player_position[1] + 1)
            new_player_position = (new_player_position[0] + 1, new_player_position[1] + 1)
            # altrimenti vai a S -> S -> NE -> NE
            new_player_position = (new_player_position[0] + 1, new_player_position[1])
            new_player_position = (new_player_position[0] + 1, new_player_position[1])
            new_player_position = (new_player_position[0] - 1, new_player_position[1] + 1)
            new_player_position = (new_player_position[0] - 1, new_player_position[1] + 1)
            # vai a S -> NE -> NE
            new_player_position = (new_player_position[0] + 1, new_player_position[1])
            new_player_position = (new_player_position[0] - 1, new_player_position[1] + 1)
            new_player_position = (new_player_position[0] - 1, new_player_position[1] + 1)
            # altrimenti vai a N -> N -> SE -> SE
            new_player_position = (new_player_position[0] - 1, new_player_position[1])
            new_player_position = (new_player_position[0] - 1, new_player_position[1])
            new_player_position = (new_player_position[0] + 1, new_player_position[1] + 1)
            new_player_position = (new_player_position[0] + 1, new_player_position[1] + 1)

        elif direction[1] == "NE":
            # vai a N -> E (+ rischio blocco) -> E
            # vai a N
            new_player_position = (new_player_position[0] - 1, new_player_position[1])
            # vai a E
            new_player_position = (new_player_position[0], new_player_position[1] + 1)
            # IF (rischio blocco) THEN vai a N -> SE (rischio blocco) -> SE
                # vai a N
            new_player_position = (new_player_position[0] - 1, new_player_position[1])
                # vai a SE
            new_player_position = (new_player_position[0] + 1, new_player_position[1] + 1)
                # IF (rischio blocco) THEN
                    # CASO DA DECIDERE
                # ELSE vai a SE
            new_player_position = (new_player_position[0] + 1, new_player_position[1] + 1)
            # ELSE vai a E
            new_player_position = (new_player_position[0], new_player_position[1] + 1)
            
        elif direction[1] == "SE":
            # vai a S -> E (rischio blocco) -> E
            # vai a S
            new_player_position = (new_player_position[0] + 1, new_player_position[1])
            # vai a E
            new_player_position = (new_player_position[0], new_player_position[1] + 1)
            # IF (rischio blocco) THEN vai a S -> NE (rischio blocco) -> NE
                # vai a S
            new_player_position = (new_player_position[0] + 1, new_player_position[1])
                # vai a NE
            new_player_position = (new_player_position[0] - 1, new_player_position[1] + 1)
                # IF (rischio blocco) THEN
                    # CASO DA DECIDERE
                # ELSE vai a NE
            new_player_position = (new_player_position[0] - 1, new_player_position[1] + 1)
            # ELSE vai a E
            new_player_position = (new_player_position[0], new_player_position[1] + 1)
            
                
    """else:
        if direction[1] == "E":

        elif direction[1] == "NE":

        elif direction[1] == "SE":

        elif direction[1] == "S":

        elif direction[1] == "SW":

        elif direction[1] == "W":

        elif direction[1] == "NW":

        elif direction[1] == "N":
    """

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

def position_for_boulder_push(current_boulder_position: Tuple[int,int], new_boulder_position: Tuple[int,int]) -> Tuple[int,int]:
    """
        returns the position where the agent should be so to push a block
        :param block_position: the current position of the block
        :param new_boulder_position: the position where the block needs to be pushed
        :return: the position where the agent should be so to move the block
    """
    i, j = current_boulder_position #raw and column of the position of the block
    # map the move the boulder needs to do with the position the agent should be
    coord_map = {
        "N": (i+1,j),
        "E": (i,j-1),
        "S": (i-1,j), 
        "W": (i,j+1),
        "NE": (i+1,j-1),
        "SE": (i-1,j-1),
        "SW": (i-1,j+1),
        "NW": (i+1,j+1)
    }

    action, action_name = action_map(current_boulder_position, new_boulder_position) #check the action the boulder has to do

    return coord_map[action_name] #return the position the agent should be given the move the boulder needs to do 


# this funciton was just for testing (its body will be put in the notebook)
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
# file for the logic of the game

from typing import Tuple, List
import numpy as np
import gym
import matplotlib.pyplot as plt
import minihack
import random

from utils import *

def avoid_the_obstacle(game_map: np.ndarray, player_position: Tuple[int, int], obstacle_position: Tuple[int, int]) -> int:
    """
        manage the player stuck by an obstacle (boulder or river)
        :param game_map: the game map as a matrix
        :param player_position: the current position of the agent
        :param obstacle_position: the position of the obstacle
        :return: the result (0 for avoid the boulder, 1 for pass the river, -1 for stuck)
    """

    # get the direction the player is following
    direction = action_map(player_position, obstacle_position)
    
    prev_player_position = player_position
    new_player_position = player_position

    valid_moves = get_valid_moves(game_map, player_position, obstacle_position)


    # check if the obstacle is a river, else is a boulder
    if game_map[obstacle_position] == ord("}"):

        # the player is going East
        if direction[1] == "E":
            # vai a N (forse muro) -> SE (rischio blocco fiume) -> SE
            if [new_player_position[0] - 1, new_player_position[1]] in valid_moves:
                # vai a N
                # env.step(0)
                prev_player_position = new_player_position
                new_player_position = get_player_location
                # vai a SE
                # env.step(5)
                prev_player_position = new_player_position
                new_player_position = get_player_location
                # IF non siamo bloccati THEN -> SE
                if not is_player_same_position(new_player_position, prev_player_position):
                    # vai a SE
                    # env.step(5)
                    prev_player_position = new_player_position
                    new_player_position = get_player_location
                    
                    # vai nella nuova casella creata a SE
                    # env.step(5)
                    prev_player_position = new_player_position
                    new_player_position = get_player_location
                    if is_player_same_position(new_player_position, prev_player_position):
                        # masso affondato ricercare path alternativo
                        """
                            CASO DA DECIDERE
                        """
                    else:
                        # attraversa il fiume
                        pass_the_river(game_map, new_player_position, 1)
                        return 1
                    
                # ELSE vai a S
                # env.step(2)   
                prev_player_position = new_player_position
                new_player_position = get_player_location

            # vai a S (forse muro) -> NE (rischio blocco fiume) -> NE
            if [new_player_position[0] + 1, new_player_position[1]] in valid_moves:
                # vai a S
                # env.step(2)
                prev_player_position = new_player_position
                new_player_position = get_player_location
                # vai a NE
                # env.step(4)
                prev_player_position = new_player_position
                new_player_position = get_player_location
                # IF non siamo bloccati THEN -> NE
                if not is_player_same_position(new_player_position, prev_player_position):
                    # vai a NE
                    # env.step(4)
                    prev_player_position = new_player_position
                    new_player_position = get_player_location

                    # vai nella nuova casella creata a NE
                    # env.step(4)
                    prev_player_position = new_player_position
                    new_player_position = get_player_location
                    if is_player_same_position(new_player_position, prev_player_position):
                        # masso affondato ricercare path alternativo
                        """
                            CASO DA DECIDERE
                        """
                    else:
                        # attraversa il fiume
                        pass_the_river(game_map, new_player_position, 1)
                        return 1
            else:     
                # bisogna spostare il masso di una casella e ricalcolare il path
                """
                    CASO DA DECIDERE
                """

        elif direction[1] == "NE":
            # vai a N -> E (rischio blocco fiume) -> E
            # vai a N
            # env.step(0)
            prev_player_position = new_player_position
            new_player_position = get_player_location
            # vai a E
            # env.step(1)
            prev_player_position = new_player_position
            new_player_position = get_player_location
            # IF non siamo bloccati THEN -> E
            if not is_player_same_position(new_player_position, prev_player_position):
                # vai a E
                # env.step(1)
                prev_player_position = new_player_position
                new_player_position = get_player_location

                # vai nella nuova casella creata a E
                # env.step(1)
                prev_player_position = new_player_position
                new_player_position = get_player_location
                if is_player_same_position(new_player_position, prev_player_position):
                    # masso affondato ricercare path alternativo
                    """
                        CASO DA DECIDERE
                    """
                else:
                    # attraversa il fiume
                    pass_the_river(game_map, new_player_position, 1)
                    return 1
            
            # vai a N -> SE (rischio blocco fiume) -> SE
            # vai a N
            # env.step(0)
            prev_player_position = new_player_position
            new_player_position = get_player_location
            # vai a SE
            # env.step(5)
            prev_player_position = new_player_position
            new_player_position = get_player_location
            # IF non siamo bloccati THEN -> SE
            if not is_player_same_position(new_player_position, prev_player_position):
                # vai a SE
                # env.step(5)
                prev_player_position = new_player_position
                new_player_position = get_player_location

                # vai nella nuova casella creata a SE
                # env.step(5)
                prev_player_position = new_player_position
                new_player_position = get_player_location
                if is_player_same_position(new_player_position, prev_player_position):
                    # masso affondato ricercare path alternativo
                    """
                        CASO DA DECIDERE
                    """
                else:
                    # attraversa il fiume
                    pass_the_river(game_map, new_player_position, 1)
                    return 1
            
        elif direction[1] == "SE":
            # vai a S -> E (rischio blocco fiume) -> E
            # vai a S
            # env.step(2)
            prev_player_position = new_player_position
            new_player_position = get_player_location
            # vai a E
            # env.step(1)
            prev_player_position = new_player_position
            new_player_position = get_player_location
            # IF non siamo bloccati THEN -> E
            if not is_player_same_position(new_player_position, prev_player_position):
                # vai a E
                # env.step(1)
                prev_player_position = new_player_position
                new_player_position = get_player_location

                # vai nella nuova casella creata a E
                # env.step(1)
                prev_player_position = new_player_position
                new_player_position = get_player_location
                if is_player_same_position(new_player_position, prev_player_position):
                    # masso affondato ricercare path alternativo
                    """
                        CASO DA DECIDERE
                    """
                else:
                    # attraversa il fiume
                    pass_the_river(game_map, new_player_position, 1)
                    return 1
                
            # vai a S -> NE (rischio blocco fiume) -> NE
            # vai a S
            # env.step(2)
            prev_player_position = new_player_position
            new_player_position = get_player_location
            # vai a NE
            # env.step(4)
            prev_player_position = new_player_position
            new_player_position = get_player_location
            # IF non siamo bloccati THEN -> NE
            if not is_player_same_position(new_player_position, prev_player_position):
                # vai a NE
                # env.step(4)
                prev_player_position = new_player_position
                new_player_position = get_player_location

                # vai nella nuova casella creata a NE
                # env.step(4)
                prev_player_position = new_player_position
                new_player_position = get_player_location
                if is_player_same_position(new_player_position, prev_player_position):
                    # masso affondato ricercare path alternativo
                    """
                        CASO DA DECIDERE
                    """
                else:
                    # attraversa il fiume
                    pass_the_river(game_map, new_player_position, 1)
                    return 1
            
                
    else:
        if direction[1] == "E":
            # vai a NE (forse muro), altrimenti SE
            if [new_player_position[0] + 1, new_player_position[1] - 1] in valid_moves:
                # vai a NE
                # env.step(4)
                prev_player_position = new_player_position
                new_player_position = get_player_location
                # IF non siamo bloccati FINE, altrimenti SE
                if not is_player_same_position(new_player_position, prev_player_position):
                    # ricalcolare path migliore
                    return 0
            else:
                # vai a SE
                # env.step(5)
                prev_player_position = new_player_position
                new_player_position = get_player_location
                # IF non siamo bloccati FINE
                if not is_player_same_position(new_player_position, prev_player_position):
                    # ricalcolare path migliore
                    return 0
            
            # non ci siamo spostati
            """
                CASO DA DECIDERE
            """

        elif direction[1] == "S":
            # vai a SE (forse muro), altrimenti SW
            if [new_player_position[0] - 1, new_player_position[1] - 1] in valid_moves:
                # vai a SE
                # env.step(5)
                prev_player_position = new_player_position
                new_player_position = get_player_location
                # IF non siamo bloccati FINE, altrimenti SW
                if not is_player_same_position(new_player_position, prev_player_position):
                    # ricalcolare path migliore
                    return 0
            else:
                # vai a SW
                # env.step(6)
                prev_player_position = new_player_position
                new_player_position = get_player_location
                # IF non siamo bloccati FINE
                if not is_player_same_position(new_player_position, prev_player_position):
                    # ricalcolare path migliore
                    return 0
            
            # non ci siamo spostati
            """
                CASO DA DECIDERE
            """
        
        elif direction[1] == "W":
            # vai a SW (forse muro), altrimenti NW
            if [new_player_position[0] - 1, new_player_position[1] + 1] in valid_moves:
                # vai a SW
                # env.step(6)
                prev_player_position = new_player_position
                new_player_position = get_player_location
                # IF non siamo bloccati FINE, altrimenti NW
                if not is_player_same_position(new_player_position, prev_player_position):
                    # ricalcolare path migliore
                    return 0
            else:
                # vai a NW
                # env.step(7)
                prev_player_position = new_player_position
                new_player_position = get_player_location
                # IF non siamo bloccati FINE
                if not is_player_same_position(new_player_position, prev_player_position):
                    # ricalcolare path migliore
                    return 0
            
            # non ci siamo spostati
            """
                CASO DA DECIDERE
            """

        elif direction[1] == "N":
            # vai a NW (forse muro), altrimenti NE
            if [new_player_position[0] + 1, new_player_position[1] + 1] in valid_moves:
                # vai a NW
                # env.step(7)
                prev_player_position = new_player_position
                new_player_position = get_player_location
                # IF non siamo bloccati FINE, altrimenti NE
                if not is_player_same_position(new_player_position, prev_player_position):
                    # ricalcolare path migliore
                    return 0
            else:
                # vai a NE
                # env.step(4)
                prev_player_position = new_player_position
                new_player_position = get_player_location
                # IF non siamo bloccati FINE
                if not is_player_same_position(new_player_position, prev_player_position):
                    # ricalcolare path migliore
                    return 0
            
            # non ci siamo spostati
            """
                CASO DA DECIDERE
            """

        elif direction[1] == "NE":
            # vai a N (forse muro), altrimenti E
            if [new_player_position[0] + 1, new_player_position[1]] in valid_moves:
                # vai a N
                # env.step(0)
                prev_player_position = new_player_position
                new_player_position = get_player_location
                # IF non siamo bloccati FINE, altrimenti E
                if not is_player_same_position(new_player_position, prev_player_position):
                    # ricalcolare path migliore
                    return 0
            else:
                # vai a E
                # env.step(1)
                prev_player_position = new_player_position
                new_player_position = get_player_location
                # IF non siamo bloccati FINE
                if not is_player_same_position(new_player_position, prev_player_position):
                    # ricalcolare path migliore
                    return 0
            
            # non ci siamo spostati
            """
                CASO DA DECIDERE
            """

        elif direction[1] == "SE":
            # vai a S (forse muro), altrimenti E
            if [new_player_position[0] - 1, new_player_position[1]] in valid_moves:
                # vai a S
                # env.step(2)
                prev_player_position = new_player_position
                new_player_position = get_player_location
                # IF non siamo bloccati FINE, altrimenti E
                if not is_player_same_position(new_player_position, prev_player_position):
                    # ricalcolare path migliore
                    return 0
            else:
                # vai a E
                # env.step(1)
                prev_player_position = new_player_position
                new_player_position = get_player_location
                # IF non siamo bloccati FINE
                if not is_player_same_position(new_player_position, prev_player_position):
                    # ricalcolare path migliore
                    return 0
            
            # non ci siamo spostati
            """
                CASO DA DECIDERE
            """

        elif direction[1] == "SW":
            # vai a S (forse muro), altrimenti W
            if [new_player_position[0] - 1, new_player_position[1]] in valid_moves:
                # vai a S
                # env.step(2)
                prev_player_position = new_player_position
                new_player_position = get_player_location
                # IF non siamo bloccati FINE, altrimenti W
                if not is_player_same_position(new_player_position, prev_player_position):
                    # ricalcolare path migliore
                    return 0
            else:
                # vai a W
                # env.step(3)
                prev_player_position = new_player_position
                new_player_position = get_player_location
                # IF non siamo bloccati FINE
                if not is_player_same_position(new_player_position, prev_player_position):
                    # ricalcolare path migliore
                    return 0
            
            # non ci siamo spostati
            """
                CASO DA DECIDERE
            """

        elif direction[1] == "NW":
            # vai a N (forse muro), altrimenti W
            if [new_player_position[0] + 1, new_player_position[1]] in valid_moves:
                # vai a N
                # env.step(0)
                prev_player_position = new_player_position
                new_player_position = get_player_location
                # IF non siamo bloccati FINE, altrimenti W
                if not is_player_same_position(new_player_position, prev_player_position):
                    # ricalcolare path migliore
                    return 0
            else:
                # vai a W
                # env.step(3)
                prev_player_position = new_player_position
                new_player_position = get_player_location
                # IF non siamo bloccati FINE
                if not is_player_same_position(new_player_position, prev_player_position):
                    # ricalcolare path migliore
                    return 0
            
            # non ci siamo spostati
            """
                CASO DA DECIDERE
            """

    # player is still stuck
    return -1



def is_player_same_position(now_position: Tuple[int, int], prev_position: Tuple[int, int]) -> bool:
    """
        checks if the player is in the same position
        :param now_position: the current position of the player
        :param prev_position: the previous position of the player 
    """
    return now_position == prev_position



def pass_the_river(game_map: np.ndarray, player_position: Tuple[int, int], direction: int):
    """
        moves the player so to pass the river
        :param game_map: the game map as a matrix
        :param player_position: the current position of the agent
        :param direction: the direction to pass the river
    """

    prev_player_position = player_position
    new_player_position = player_position

    while is_player_same_position(new_player_position, prev_player_position):
        # env.step(direction)
        prev_player_position = new_player_position
        new_player_position = get_player_location
    


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
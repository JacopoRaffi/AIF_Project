# file for the logic of the game

from typing import Tuple, List
import numpy as np
import gym
import matplotlib.pyplot as plt
import minihack

from utils import *

def avoid_the_obstacle(game_map: np.ndarray, player_position: Tuple[int, int], direction: int, env: gym.Env) -> int:
    """
        manage the player stuck by an obstacle (boulder or river)
        :param game_map: the game map as a matrix
        :param player_position: the current position of the agent
        :param direction: the direction the agent is going
        :param env: the environment of the game
        :return: the result (0 find path for the boulder, 1 find path for the stairs, -1 error)
    """
    
    obstacle_position = get_obstacle_location(player_position, direction)

    prev_player_position = player_position
    new_player_position = player_position

    valid_moves = get_valid_moves(game_map, player_position, obstacle_position, True)

    # check if the obstacle is a river, else is a boulder
    if game_map[obstacle_position] == ord("}"):

        # the player is stuck going to East (river)
        if direction == 0:
            # vai a N (forse muro) -> SE (rischio blocco fiume) -> SE
            if (new_player_position[0] - 1, new_player_position[1]) in valid_moves:
                # vai a N
                state,_,_,_ = env.step(0)
                prev_player_position = new_player_position
                new_player_position = get_player_location(state["chars"])
                # vai a SE
                state,_,_,_ = env.step(5)
                prev_player_position = new_player_position
                new_player_position = get_player_location(state["chars"])
                # IF non siamo bloccati THEN -> SE
                if not is_player_same_position(new_player_position, prev_player_position):
                    # vai a SE
                    state,_,_,_ = env.step(5)
                    prev_player_position = new_player_position
                    new_player_position = get_player_location(state["chars"])
  
                    if is_player_same_position(new_player_position, prev_player_position):
                        # masso affondato - cercare path alternativo
                        return 0
                    else:
                        # creata nuova casella - cercare path per raggiungere le scale
                        return 1
                    
                # ELSE vai a S (torno a posizione iniziale)
                state,_,_,_ = env.step(2)   
                prev_player_position = new_player_position
                new_player_position = get_player_location(state["chars"])

            # vai a S (forse muro) -> NE (rischio blocco fiume) -> NE
            if (new_player_position[0] + 1, new_player_position[1]) in valid_moves:
                # vai a S
                state,_,_,_ = env.step(2)
                prev_player_position = new_player_position
                new_player_position = get_player_location(state["chars"])
                # vai a NE
                state,_,_,_ = env.step(4)
                prev_player_position = new_player_position
                new_player_position = get_player_location(state["chars"])
                # IF non siamo bloccati THEN -> NE
                if not is_player_same_position(new_player_position, prev_player_position):
                    # vai a NE
                    state,_,_,_ = env.step(4)
                    prev_player_position = new_player_position
                    new_player_position = get_player_location(state["chars"])
                    if is_player_same_position(new_player_position, prev_player_position):
                        # masso affondato - cercare path alternativo
                        return 0
                    else:
                        # creata nuova casella - cercare path per raggiungere le scale
                        return 1
                
                # ELSE vai a N (torno a posizione iniziale)
                state,_,_,_ = env.step(0)
                prev_player_position = new_player_position
                new_player_position = get_player_location(state["chars"])
                 
            # bisogna spostare il masso di una casella e ricalcolare il path
            if (new_player_position[0] - 1, new_player_position[1] + 1) in valid_moves and (new_player_position[0] + 1, new_player_position[1]) in valid_moves:
                # vai a NE
                state,_,_,_ = env.step(4)
                prev_player_position = new_player_position
                new_player_position = get_player_location(state["chars"])
                # vai a S
                state,_,_,_ = env.step(2)
                prev_player_position = new_player_position
                new_player_position = get_player_location(state["chars"])

                return 0
        

        # the player is stuck going to North-East (river)
        elif direction == 4:
            # vai a N -> E (rischio blocco fiume) -> E
            # vai a N
            state,_,_,_ = env.step(0)
            prev_player_position = new_player_position
            new_player_position = get_player_location(state["chars"])
            # vai a E
            state,_,_,_ = env.step(1)
            prev_player_position = new_player_position
            new_player_position = get_player_location(state["chars"])
            # IF non siamo bloccati THEN -> E
            if not is_player_same_position(new_player_position, prev_player_position):
                # vai a E
                # env.step(1)
                prev_player_position = new_player_position
                new_player_position = get_player_location
                
                if is_player_same_position(new_player_position, prev_player_position):
                    # masso affondato - cercare path alternativo
                    return 0
                else:
                    # creata nuova casella - cercare path per raggiungere le scale
                    return 1
            
            # vai a N -> SE (rischio blocco fiume) -> SE
            # vai a N
            state,_,_,_ = env.step(0)
            prev_player_position = new_player_position
            new_player_position = get_player_location(state["chars"])
            # vai a SE
            state,_,_,_ = env.step(5)
            prev_player_position = new_player_position
            new_player_position = get_player_location(state["chars"])
            # IF non siamo bloccati THEN -> SE
            if not is_player_same_position(new_player_position, prev_player_position):
                # vai a SE
                # env.step(5)
                prev_player_position = new_player_position
                new_player_position = get_player_location

                if is_player_same_position(new_player_position, prev_player_position):
                    # masso affondato - cercare path alternativo
                    return 0
                else:
                    # creata nuova casella - cercare path per raggiungere le scale
                    return 1
            
            # vai a S
            state,_,_,_ = env.step(2)
            prev_player_position = new_player_position
            new_player_position = get_player_location(state["chars"])

            valid_moves = get_valid_moves(state["chars"], new_player_position, obstacle_position, True)
            # bisogna spostare il masso di una casella e ricalcolare il path
            if (new_player_position[0] - 1, new_player_position[1] + 1) in valid_moves and (new_player_position[0] + 1, new_player_position[1]) in valid_moves:
                # vai a NE
                state,_,_,_ = env.step(4)
                prev_player_position = new_player_position
                new_player_position = get_player_location(state["chars"])
                # vai a S
                state,_,_,_ = env.step(2)
                prev_player_position = new_player_position
                new_player_position = get_player_location(state["chars"])

                return 0


        # the player is stuck going to South-East (river)
        elif direction == 5:
            # vai a S -> E (rischio blocco fiume) -> E
            # vai a S
            state,_,_,_ = env.step(2)
            prev_player_position = new_player_position
            new_player_position = get_player_location(state["chars"])
            # vai a E
            state,_,_,_ = env.step(1)
            prev_player_position = new_player_position
            new_player_position = get_player_location(state["chars"])
            # IF non siamo bloccati THEN -> E
            if not is_player_same_position(new_player_position, prev_player_position):
                # vai a E
                # env.step(1)
                prev_player_position = new_player_position
                new_player_position = get_player_location

                if is_player_same_position(new_player_position, prev_player_position):
                    # masso affondato - cercare path alternativo
                    return 0
                else:
                    # creata nuova casella - cercare path per raggiungere le scale
                    return 1
                
            # vai a S -> NE (rischio blocco fiume) -> NE
            # vai a S
            state,_,_,_ = env.step(2)
            prev_player_position = new_player_position
            new_player_position = get_player_location(state["chars"])
            # vai a NE
            state,_,_,_ = env.step(4)
            prev_player_position = new_player_position
            new_player_position = get_player_location(state["chars"])
            # IF non siamo bloccati THEN -> NE
            if not is_player_same_position(new_player_position, prev_player_position):
                # vai a NE
                state,_,_,_ = env.step(4)
                prev_player_position = new_player_position
                new_player_position = get_player_location(state["chars"])

                if is_player_same_position(new_player_position, prev_player_position):
                    # masso affondato - cercare path alternativo
                    return 0
                else:
                    # creata nuova casella - cercare path per raggiungere le scale
                    return 1
            
            # vai a N
            state,_,_,_ = env.step(0)
            prev_player_position = new_player_position
            new_player_position = get_player_location(state["chars"])

            valid_moves = get_valid_moves(state["chars"], new_player_position, obstacle_position, True)
            # bisogna spostare il masso di una casella e ricalcolare il path
            if (new_player_position[0] - 1, new_player_position[1] + 1) in valid_moves and (new_player_position[0] + 1, new_player_position[1]) in valid_moves:
                # vai a NE
                state,_,_,_ = env.step(4)
                prev_player_position = new_player_position
                new_player_position = get_player_location(state["chars"])
                # vai a S
                state,_,_,_ = env.step(2)
                prev_player_position = new_player_position
                new_player_position = get_player_location(state["chars"])

                return 0
                


    else:
        # the player is stuck going to East (boulder)
        if direction == 1:
            # vai a NE (forse muro), altrimenti SE
            if (new_player_position[0] - 1, new_player_position[1] + 1) in valid_moves:
                # vai a NE
                state,_,_,_ = env.step(4)
                prev_player_position = new_player_position
                new_player_position = get_player_location(state["chars"])
                # IF non siamo bloccati FINE, altrimenti SE
                if not is_player_same_position(new_player_position, prev_player_position):
                    # ricalcolare path migliore
                    return 0
            
            # vai a SE
            state,_,_,_ = env.step(5)
            prev_player_position = new_player_position
            new_player_position = get_player_location(state["chars"])
            # IF non siamo bloccati FINE
            if not is_player_same_position(new_player_position, prev_player_position):
                # ricalcolare path migliore
                return 0
            
            # non ci siamo spostati
            """
                CASO DA DECIDERE
            """
        
        # the player is stuck going to South (boulder)
        elif direction == 2:
            # vai a SE (forse muro), altrimenti SW
            if (new_player_position[0] + 1, new_player_position[1] + 1) in valid_moves:
                # vai a SE
                state,_,_,_ = env.step(5)
                prev_player_position = new_player_position
                new_player_position = get_player_location(state["chars"])
                # IF non siamo bloccati FINE, altrimenti SW
                if not is_player_same_position(new_player_position, prev_player_position):
                    # ricalcolare path migliore
                    return 0
            
            # vai a SW
            state,_,_,_ = env.step(6)
            prev_player_position = new_player_position
            new_player_position = get_player_location(state["chars"])
            # IF non siamo bloccati FINE
            if not is_player_same_position(new_player_position, prev_player_position):
                # ricalcolare path migliore
                return 0
            
            # non ci siamo spostati
            """
                CASO DA DECIDERE
            """
        
        # the player is stuck going to West (boulder)
        elif direction == 3:
            # vai a SW (forse muro), altrimenti NW
            if (new_player_position[0] + 1, new_player_position[1] - 1) in valid_moves:
                # vai a SW
                state,_,_,_ = env.step(6)
                prev_player_position = new_player_position
                new_player_position = get_player_location(state["chars"])
                # IF non siamo bloccati FINE, altrimenti NW
                if not is_player_same_position(new_player_position, prev_player_position):
                    # ricalcolare path migliore
                    return 0
            
            # vai a NW
            state,_,_,_ = env.step(7)
            prev_player_position = new_player_position
            new_player_position = get_player_location(state["chars"])
            # IF non siamo bloccati FINE
            if not is_player_same_position(new_player_position, prev_player_position):
                # ricalcolare path migliore
                return 0
            
            # non ci siamo spostati
            """
                CASO DA DECIDERE
            """
        
        # the player is stuck going to North (boulder)
        elif direction == 0:
            # vai a NW (forse muro), altrimenti NE
            if (new_player_position[0] - 1, new_player_position[1] - 1) in valid_moves:
                # vai a NW
                state,_,_,_ = env.step(7)
                prev_player_position = new_player_position
                new_player_position = get_player_location(state["chars"])
                # IF non siamo bloccati FINE, altrimenti NE
                if not is_player_same_position(new_player_position, prev_player_position):
                    # ricalcolare path migliore
                    return 0
            
            # vai a NE
            state,_,_,_ = env.step(4)
            prev_player_position = new_player_position
            new_player_position = get_player_location(state["chars"])
            # IF non siamo bloccati FINE
            if not is_player_same_position(new_player_position, prev_player_position):
                # ricalcolare path migliore
                return 0
            
            # non ci siamo spostati
            """
                CASO DA DECIDERE
            """
        
        # the player is stuck going to North-East (boulder)
        elif direction == 4:
            # vai a N
            state,_,_,_ = env.step(0)
            prev_player_position = new_player_position
            new_player_position = get_player_location(state["chars"])
            # IF non siamo bloccati FINE, altrimenti E
            if not is_player_same_position(new_player_position, prev_player_position):
                # ricalcolare path migliore
                return 0
            else:
                # vai a E
                state,_,_,_ = env.step(1)
                prev_player_position = new_player_position
                new_player_position = get_player_location(state["chars"])
                # IF non siamo bloccati FINE
                if not is_player_same_position(new_player_position, prev_player_position):
                    # ricalcolare path migliore
                    return 0
            
            # non ci siamo spostati
            """
                CASO DA DECIDERE
            """
        
        # the player is stuck going to South-East (boulder)
        elif direction == 5:
            # vai a S
            state,_,_,_ = env.step(2)
            prev_player_position = new_player_position
            new_player_position = get_player_location(state["chars"])
            # IF non siamo bloccati FINE, altrimenti E
            if not is_player_same_position(new_player_position, prev_player_position):
                # ricalcolare path migliore
                return 0
            else:
                # vai a E
                state,_,_,_ = env.step(1)
                prev_player_position = new_player_position
                new_player_position = get_player_location(state["chars"])
                # IF non siamo bloccati FINE
                if not is_player_same_position(new_player_position, prev_player_position):
                    # ricalcolare path migliore
                    return 0
                
            # non ci siamo spostati
            """
                CASO DA DECIDERE
            """
        
        # the player is stuck going to South-West (boulder)
        elif direction == 6:
            # vai a S
            state,_,_,_ = env.step(2)
            prev_player_position = new_player_position
            new_player_position = get_player_location(state["chars"])
            # IF non siamo bloccati FINE, altrimenti W
            if not is_player_same_position(new_player_position, prev_player_position):
                # ricalcolare path migliore
                return 0
            else:
                # vai a W
                state,_,_,_ = env.step(3)
                prev_player_position = new_player_position
                new_player_position = get_player_location(state["chars"])
                # IF non siamo bloccati FINE
                if not is_player_same_position(new_player_position, prev_player_position):
                    # ricalcolare path migliore
                    return 0
                
            # non ci siamo spostati
            """
                CASO DA DECIDERE
            """
        
        # the player is stuck going to North-West (boulder)
        elif direction == 7:
            # vai a N
            state,_,_,_ = env.step(0)
            prev_player_position = new_player_position
            new_player_position = get_player_location(state["chars"])
            # IF non siamo bloccati FINE, altrimenti W
            if not is_player_same_position(new_player_position, prev_player_position):
                # ricalcolare path migliore
                return 0
            else:
                # vai a W
                state,_,_,_ = env.step(3)
                prev_player_position = new_player_position
                new_player_position = get_player_location(state["chars"])
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

def get_obstacle_location(player_position: Tuple[int, int], direction: int) -> Tuple[int, int]:
    """
        returns the position of the obstacle in front of the agent
        :param player_position: the current position of the agent
        :param direction: the direction the agent is going
        :return: the position of the obstacle
    """

    obstacle_position = (-1, -1)

    # i is raw, j is column of matrix
    i, j = player_position[0], player_position[1]
    obstacle_map = {
        # North
        0: (i-2, j),
        # East
        1: (i, j+2),
        # South
        2: (i+2, j),
        # West
        3: (i, j-2),
        # North-East
        4: (i-2, j+2),
        # South-East
        5: (i+2, j+2),
        # South-West
        6: (i+2, j-2),
        # North-West
        7: (i-2, j-2)
    }

    obstacle_position = obstacle_map[direction]

    return obstacle_position

def is_player_same_position(now_position: Tuple[int, int], prev_position: Tuple[int, int]) -> bool:
    """
        checks if the player is in the same position
        :param now_position: the current position of the player
        :param prev_position: the previous position of the player 
    """
    return now_position == prev_position

def position_for_boulder_push(current_boulder_position: Tuple[int,int], new_boulder_position: Tuple[int,int]) -> Tuple[int, Tuple[int,int]]:
    """
        returns the position where the agent should be so to push a block
        :param block_position: the current position of the block
        :param new_boulder_position: the position where the block needs to be pushed, given the river path is the second element of it
        :return: the action the agent needs to perform and the position where the agent should be so to move the block
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

    return action, coord_map[action_name] #return the position the agent should be given the move the boulder needs to do 

def push_boulder_path(boulder_path: List[Tuple[int, int]]) -> Tuple[List[int], List[Tuple[int,int]]]:
    """
        returns the path the agent follows to push a boulder along its(of the boulder) path
        :param boulder_path: the path the boulder has to follow
        :return: two lists, one for the actions of the agent and one for the positions the agent should follow
    """ 

    agent_path = []

    for i in range(len(boulder_path)-1): 
        # boulder_path[i] current position of the boulder
        # boulder_path[i+1] new position of the boulder
        _,agent_position = position_for_boulder_push(boulder_path[i], boulder_path[i+1]) # get the action and the position the agent should be
        
        # append where the agent should be and his new position (it will be the same as the boulder before the move)
        #print(agent_position, boulder_path[i])
        if not agent_position == boulder_path[i-1]:
            agent_path.extend([agent_position, boulder_path[i]])
        else:
            agent_path.append(boulder_path[i])


    #(agent_path)
    if len(agent_path) > 0:
        agent_actions,names = actions_from_path(agent_path[0], agent_path[1:]) #get the actions the agent should perform to follow the path
    
    #print(names, "E")
    #agent_actions.append(1) #add action to push to the river
    return agent_actions, agent_path

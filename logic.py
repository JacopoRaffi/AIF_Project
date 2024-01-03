# file for the logic of the game

from typing import Tuple, List
import numpy as np
import gym
import matplotlib.pyplot as plt
import minihack
import random

from utils import *

def avoid_obstacle(game_map: np.ndarray, player_position: Tuple[int, int], direction: int, env: gym.Env) -> Tuple[gym.Env, int, Tuple[int,int]]:
    """
        Manage the player stuck by an obstacle (boulder or river)

        Parameters:
        - game_map (np.ndarray): the game map as a matrix
        - player_position (Tuple[int, int]): the current position of the agent
        - direction (int): the direction the agent is going
        - env (gym.Env): the environment of the game

        Returns:
        - state (gym.Env): the new state of the game
        - state_type (int): 0 (search new path player->boulder->path)
                            or 1 (search the new path player->river)
                            or 2 (search new path player->boulder->path and add boulder in blacklist)
        - river_target (Tuple[int,int]): the target of the river (if it's a river cell)
    """
    
    obstacle_position = get_obstacle_location(player_position, direction)

    prev_player_position = player_position
    new_player_position = player_position

    valid_moves = get_valid_moves(game_map, player_position, obstacle_position, True)

    state = None
    river_target = (-1,-1)

    # Check if the obstacle is a river, else is a boulder
    if game_map[obstacle_position] == ord("}"):
        # The player is stuck going to East (river)
        if direction == 0:
            # Go to N (maybe wall) -> SE (maybe river osbtacle)
            if (new_player_position[0] - 1, new_player_position[1]) in valid_moves:
                # Go to N
                state,_,_,_ = env.step(0)
                prev_player_position = new_player_position
                new_player_position = get_player_location(state["chars"])
                # Go to SE
                state,_,_,_ = env.step(5)
                prev_player_position = new_player_position
                new_player_position = get_player_location(state["chars"])
                # IF we are not stuck
                if not is_player_same_position(new_player_position, prev_player_position):
                    if state["chars"][(new_player_position[0] + 1, new_player_position[1] + 1)] == ord("}"):
                        # boulder sunk - search alternative path
                        return state, 0, river_target
                    else:
                        # new cell created - search path to reach the stairs
                        river_target = (new_player_position[0] + 1, new_player_position[1] + 1)
                        return state, 1, river_target
                
                # Go to S (return to initial position)
                state,_,_,_ = env.step(2)   
                prev_player_position = new_player_position
                new_player_position = get_player_location(state["chars"])

            else:
                # We are stuck - boulder in blacklist
                return state, 2, river_target

            # Go to S (maybe wall) -> NE (maybe river osbtacle)
            if (new_player_position[0] + 1, new_player_position[1]) in valid_moves:
                # Go to S
                state,_,_,_ = env.step(2)
                prev_player_position = new_player_position
                new_player_position = get_player_location(state["chars"])
                # Go to NE
                state,_,_,_ = env.step(4)
                prev_player_position = new_player_position
                new_player_position = get_player_location(state["chars"])
                # IF we are not stuck
                if not is_player_same_position(new_player_position, prev_player_position):
                    if state["chars"][(new_player_position[0] - 1, new_player_position[1] + 1)] == ord("}"):
                        # boulder sunk - search alternative path
                        return state, 0, river_target
                    else:
                        # new cell created - search path to reach the stairs
                        river_target = (new_player_position[0] - 1, new_player_position[1] + 1)
                        return state, 1, river_target
                
                # Go to N (return to initial position)
                state,_,_,_ = env.step(0)
                prev_player_position = new_player_position
                new_player_position = get_player_location(state["chars"])

            else:
                # We are stuck - boulder in blacklist
                return state, 2, river_target  

            # Move the boulder of one cell and recalculate the path   
            if (new_player_position[0] - 1, new_player_position[1] + 1) in valid_moves and (new_player_position[0] + 1, new_player_position[1]) in valid_moves:
                # Go to NE
                state,_,_,_ = env.step(4)
                prev_player_position = new_player_position
                new_player_position = get_player_location(state["chars"])
                # Go to S
                state,_,_,_ = env.step(2)
                prev_player_position = new_player_position
                new_player_position = get_player_location(state["chars"])
                # Find new path
                return state, 0, river_target
        

        # The player is stuck going to North-East (river)
        elif direction == 4:
            # Go to N (maybe wall) -> E (maybe river osbtacle)
            # Go to N
            state,_,_,_ = env.step(0)
            prev_player_position = new_player_position
            new_player_position = get_player_location(state["chars"])
            # Go to E
            state,_,_,_ = env.step(1)
            prev_player_position = new_player_position
            new_player_position = get_player_location(state["chars"])
            # IF we are not stuck
            if not is_player_same_position(new_player_position, prev_player_position):
                if state["chars"][(new_player_position[0], new_player_position[1] + 1)] == ord("}"):
                    # boulder sunk - search alternative path
                    return state, 0, river_target
                else:
                    # new cell created - search path to reach the stairs
                    river_target = (new_player_position[0], new_player_position[1] + 1)
                    return state, 1, river_target
            
            # Go to N -> SE (maybe river osbtacle)
            # Go to N
            state,_,_,_ = env.step(0)
            prev_player_position = new_player_position
            new_player_position = get_player_location(state["chars"])
            # Go to SE
            state,_,_,_ = env.step(5)
            prev_player_position = new_player_position
            new_player_position = get_player_location(state["chars"])
            # IF we are not stuck
            if not is_player_same_position(new_player_position, prev_player_position):
                if state["chars"][(new_player_position[0] + 1, new_player_position[1] + 1)] == ord("}"):
                    # boulder sunk - search alternative path
                    return state, 0, river_target
                else:
                    # new cell created - search path to reach the stairs
                    river_target = (new_player_position[0] + 1, new_player_position[1] + 1)
                    return state, 1, river_target
            
            # Go to S
            state,_,_,_ = env.step(2)
            prev_player_position = new_player_position
            new_player_position = get_player_location(state["chars"])

            valid_moves = get_valid_moves(state["chars"], new_player_position, obstacle_position, True)
            # Move the boulder of one cell and recalculate the path
            if (new_player_position[0] - 1, new_player_position[1] + 1) in valid_moves and (new_player_position[0] + 1, new_player_position[1]) in valid_moves:
                # Go to NE
                state,_,_,_ = env.step(4)
                prev_player_position = new_player_position
                new_player_position = get_player_location(state["chars"])
                # Go to S
                state,_,_,_ = env.step(2)
                prev_player_position = new_player_position
                new_player_position = get_player_location(state["chars"])
                # Find new path
                return state, 0, river_target


        # The player is stuck going to South-East (river)
        elif direction == 5:
            # Go to S -> E (maybe river osbtacle)
            # Go to S
            state,_,_,_ = env.step(2)
            prev_player_position = new_player_position
            new_player_position = get_player_location(state["chars"])
            # Go to E
            state,_,_,_ = env.step(1)
            prev_player_position = new_player_position
            new_player_position = get_player_location(state["chars"])
            # IF we are not stuck
            if not is_player_same_position(new_player_position, prev_player_position):
                if state["chars"][(new_player_position[0], new_player_position[1] + 1)] == ord("}"):
                    # boulder sunk - search alternative path
                    return state, 0, river_target
                else:
                    # new cell created - search path to reach the stairs
                    river_target = (new_player_position[0], new_player_position[1] + 1)
                    return state, 1, river_target

            # Go to S -> NE (maybe river osbtacle)
            # Go to S
            state,_,_,_ = env.step(2)
            prev_player_position = new_player_position
            new_player_position = get_player_location(state["chars"])
            # Go to NE
            state,_,_,_ = env.step(4)
            prev_player_position = new_player_position
            new_player_position = get_player_location(state["chars"])
            # IF we are not stuck
            if not is_player_same_position(new_player_position, prev_player_position):
                if state["chars"][(new_player_position[0] - 1, new_player_position[1] + 1)] == ord("}"):
                    # boulder sunk - search alternative path
                    return state, 0, river_target
                else:
                    # new cell created - search path to reach the stairs
                    river_target = (new_player_position[0] - 1, new_player_position[1] + 1)
                    return state, 1, river_target
            
            # Go to N
            state,_,_,_ = env.step(0)
            prev_player_position = new_player_position
            new_player_position = get_player_location(state["chars"])

            valid_moves = get_valid_moves(state["chars"], new_player_position, obstacle_position, True)
            # Move the boulder of one cell and recalculate the path
            if (new_player_position[0] - 1, new_player_position[1] + 1) in valid_moves and (new_player_position[0] + 1, new_player_position[1]) in valid_moves:
                # Go to NE
                state,_,_,_ = env.step(4)
                prev_player_position = new_player_position
                new_player_position = get_player_location(state["chars"])
                # Go to S
                state,_,_,_ = env.step(2)
                prev_player_position = new_player_position
                new_player_position = get_player_location(state["chars"])
                # Find new path
                return state, 0, river_target
    


    else:
        # The player is stuck going to East (boulder)
        if direction == 1:
            # Go to NE (maybe wall), else SE
            if (new_player_position[0] - 1, new_player_position[1] + 1) in valid_moves:
                # Go to NE
                state,_,_,_ = env.step(4)
                prev_player_position = new_player_position
                new_player_position = get_player_location(state["chars"])
                # If we are not stuck END, else SE
                if not is_player_same_position(new_player_position, prev_player_position):
                    # Find new path
                    return state, 0, river_target
            
            # Go to SE
            state,_,_,_ = env.step(5)
            prev_player_position = new_player_position
            new_player_position = get_player_location(state["chars"])

        
        # The player is stuck going to South (boulder)
        elif direction == 2:
            # Go to SE (maybe wall), else SW
            if (new_player_position[0] + 1, new_player_position[1] + 1) in valid_moves:
                # Go to SE
                state,_,_,_ = env.step(5)
                prev_player_position = new_player_position
                new_player_position = get_player_location(state["chars"])
                # If we are not stuck END, else SW
                if not is_player_same_position(new_player_position, prev_player_position):
                    # Find new path
                    return state, 0, river_target
            
            # Go to SW
            state,_,_,_ = env.step(6)
            prev_player_position = new_player_position
            new_player_position = get_player_location(state["chars"])
            
        
        # The player is stuck going to West (boulder)
        elif direction == 3:
            # Go to SW (maybe wall), else NW
            if (new_player_position[0] + 1, new_player_position[1] - 1) in valid_moves:
                # Go to SW
                state,_,_,_ = env.step(6)
                prev_player_position = new_player_position
                new_player_position = get_player_location(state["chars"])
                # IF we are not stuck END, else NW
                if not is_player_same_position(new_player_position, prev_player_position):
                    # Find new path
                    return state, 0, river_target
            
            # Go to NW
            state,_,_,_ = env.step(7)
            prev_player_position = new_player_position
            new_player_position = get_player_location(state["chars"])
            
        
        # The player is stuck going to North (boulder)
        elif direction == 0:
            # Go to NW (maybe wall), else NE
            if (new_player_position[0] - 1, new_player_position[1] - 1) in valid_moves:
                # Go to NW
                state,_,_,_ = env.step(7)
                prev_player_position = new_player_position
                new_player_position = get_player_location(state["chars"])
                # IF we are not stuck END, else NE
                if not is_player_same_position(new_player_position, prev_player_position):
                    # Find new path
                    return state, 0, river_target
            
            # Go to NE
            state,_,_,_ = env.step(4)
            prev_player_position = new_player_position
            new_player_position = get_player_location(state["chars"])
            
        
        # The player is stuck going to North-East (boulder)
        elif direction == 4:
            # Go to N
            state,_,_,_ = env.step(0)
            prev_player_position = new_player_position
            new_player_position = get_player_location(state["chars"])
            # IF we are stuck E
            if is_player_same_position(new_player_position, prev_player_position):
                # Go to E
                state,_,_,_ = env.step(1)
                prev_player_position = new_player_position
                new_player_position = get_player_location(state["chars"])
                
        
        # The player is stuck going to South-East (boulder)
        elif direction == 5:
            # Go to S
            state,_,_,_ = env.step(2)
            prev_player_position = new_player_position
            new_player_position = get_player_location(state["chars"])
            # IF we are stuck E
            if is_player_same_position(new_player_position, prev_player_position):
                # Go to E
                state,_,_,_ = env.step(1)
                prev_player_position = new_player_position
                new_player_position = get_player_location(state["chars"])
                
        
        # The player is stuck going to South-West (boulder)
        elif direction == 6:
            # Go to S
            state,_,_,_ = env.step(2)
            prev_player_position = new_player_position
            new_player_position = get_player_location(state["chars"])
            # IF we are stuck W
            if is_player_same_position(new_player_position, prev_player_position):
                # Go to W
                state,_,_,_ = env.step(3)
                prev_player_position = new_player_position
                new_player_position = get_player_location(state["chars"])
                
        
        # The player is stuck going to North-West (boulder)
        elif direction == 7:
            # Go to N
            state,_,_,_ = env.step(0)
            prev_player_position = new_player_position
            new_player_position = get_player_location(state["chars"])
            # IF we are stuck W
            if is_player_same_position(new_player_position, prev_player_position):
                # Go to W
                state,_,_,_ = env.step(3)
                prev_player_position = new_player_position
                new_player_position = get_player_location(state["chars"])

    # Find new path
    return state, 0, river_target

def get_obstacle_location(player_position: Tuple[int, int], direction: int) -> Tuple[int, int]:
    """
        Returns the position of the obstacle in front of the agent

        Parameters:
        - player_position (Tuple[int, int]): the current position of the agent
        - direction (int): the direction the agent is going

        Returns:
        - obstacle_position (Tuple[int, int]): the position of the obstacle
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
        Checks if the player is in the same position

        Parameters:
        - now_position (Tuple[int, int]): the current position of the player
        - prev_position (Tuple[int, int]): the previous position of the player

        Returns:
        - (bool): True if the player is in the same position, False otherwise
    """

    return now_position == prev_position

def position_for_boulder_push(current_boulder_position: Tuple[int,int], new_boulder_position: Tuple[int,int]) -> Tuple[int, Tuple[int,int]]:
    """
        Returns the action the agent should perform and the position where the agent should be so to move the block

        Parameters:
        - current_boulder_position (Tuple[int, int]): the current position of the block
        - new_boulder_position (Tuple[int, int]): the position where the block needs to be pushed, given the river path is the second element of it

        Returns:
        - action (int): the action the agent needs to perform
        - agent_position (Tuple[int, int]): the position where the agent should be so to move the block
    """

    i, j = current_boulder_position # Raw and column of the position of the block
    # Map the move the boulder needs to do with the position the agent should be
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

    action, action_name = action_map(current_boulder_position, new_boulder_position) # Check the action the boulder has to do

    return action, coord_map[action_name] # Return the position the agent should be given the move the boulder needs to do 

def push_boulder_path(boulder_path: List[Tuple[int, int]]) -> Tuple[List[int], List[Tuple[int,int]]]:
    """
        Returns the path the agent follows to push a boulder along its(of the boulder) path

        Parameters:
        - boulder_path (List[Tuple[int, int]]): the path the boulder has to follow

        Returns:
        - agent_actions (List[int]): the actions the agent should perform to follow the path
        - agent_path (List[Tuple[int,int]]): the positions the agent should follow
    """ 

    agent_path = []

    for i in range(len(boulder_path)-1): 
        # boulder_path[i] current position of the boulder
        # boulder_path[i+1] new position of the boulder
        _,agent_position = position_for_boulder_push(boulder_path[i], boulder_path[i+1]) # Get the action and the position the agent should be
        
        # Append where the agent should be and his new position (it will be the same as the boulder before the move)
        # print(agent_position, boulder_path[i])
        if not agent_position == boulder_path[i-1]:
            agent_path.extend([agent_position, boulder_path[i]])
        else:
            agent_path.append(boulder_path[i])


    # (agent_path)
    if len(agent_path) > 0:
        agent_actions,names = actions_from_path(agent_path[0], agent_path[1:]) # Get the actions the agent should perform to follow the path
    
    return agent_actions, agent_path

def choose_best_action(valid_moves: List[Tuple[int, int]], game_map: np.ndarray, player_position: Tuple[int, int], black_list_boulder) -> int: 
    """
        Choose the best action for the agent so to find a water block (as a result, also the river)

        Parameters:
        - valid_moves (List[Tuple[int, int]]): all the moves the agent can perform from its position
        - game_map (np.ndarray): the game map as a matrix
        - player_position (Tuple[int, int]): the current position of the agent
        - black_list_boulder (List[Tuple[int, int]]): the list of the boulders that we can't move

        Returns:
        - action (int): the best action to perform
    """

    action = -1
    actions = []
    boulder_positions = get_boulder_locations(game_map, black_list_boulder)
    not_same_coords = [move for move in valid_moves if move not in boulder_positions]

    if len(not_same_coords) > 0: # If there is a step that doesn't move a boulder
        # Prioritize steps that would move the player on different raw than the boulders
        not_same_y_coords = [move for move in not_same_coords if all(move[0] != boulder[0] for boulder in boulder_positions)]

        if not len(not_same_y_coords) > 0:
            actions = [action_map(player_position, move)[0] for move in not_same_coords]
        else:
            actions = [action_map(player_position, move)[0] for move in not_same_y_coords]  

    else: # Almost impossible to happen (agent completely cornered by boulders)
        actions = [action_map(player_position, move)[0] for move in valid_moves]

    action = random.choice(actions)
    return action

def find_stairs(game_env: gym.Env, game_map: np.ndarray) -> List[Tuple[int, int]]:
    """
        Finds the stairs in the map

        Parameters:
        - game_env (gym.Env): the environment of the game
        - game_map (np.ndarray): the game map as a matrix

        Returns:
        - stairs_coordinates (List[Tuple[int, int]]): the coordinates of the stairs
    """

    stairs_coordinates = get_exit_location(game_map)
    if not stairs_coordinates is None:
        return stairs_coordinates
    else:
        return None

    found = False
    while not found: 
        action = 1
        obs_state, _, _, _ = game_env.step(action)
        game_map = obs_state["chars"] # Update observable map so to take the next "best action"
        color_map = obs_state["colors"]
        stairs_coordinates = get_exit_location(game_map)
        if not stairs_coordinates is None: # If the river is found
            found = True

    return stairs_coordinates

def find_river(game_env: gym.Env, game_map: np.ndarray, black_list_boulder) -> List[Tuple[int, int]]:
    """
        Moves the player until a water block is found.
        The river is assumed to be a vertical straight line of water blocks

        Parameters:
        - game_env (gym.Env): the environment of the game
        - game_map (np.ndarray): the game map as a matrix
        - black_list_boulder (List[Tuple[int, int]]): the list of the boulders that we can't move

        Returns:
        - river_coordinates (List[Tuple[int, int]]): the coordinates of the water blocks of the river
    """
    
    # Check if the river is in the initial map
    river_coordinates = get_river_locations(game_map)
    if len(river_coordinates) > 0:
            found = True
            return river_coordinates
    # If the river is not seen player will pefrom moves until it's found
    found = False
    player_location = get_player_location(game_map)
    while not found: 
        action = choose_best_action(get_valid_moves(game_map, player_location, (-1,-1), False, False), game_map, player_location, black_list_boulder)
        obs_state, _, _, _ = game_env.step(action)

        river_coordinates = get_river_locations(obs_state["chars"])
        if len(river_coordinates) > 0: # If the river is found
            found = True
        
        game_map = obs_state["chars"] # Update observable map so to take the next "best action"
    
    return river_coordinates
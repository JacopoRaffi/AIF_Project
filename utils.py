import numpy as np
import math

from typing import Tuple, List

def get_player_location(game_map: np.ndarray, symbol: str = "@") -> Tuple[int, int]:
    """
        gets the coordinates of the player in the game map
        :param game_map: the game map
        :param symbol: the symbol of the agent
        :return: the coordinates of the agent
    """

    x, y = np.where(game_map == ord(symbol))
    return (x[0], y[0])

def get_boulder_locationV(game_map: np.ndarray, symbol : str = "`") -> Tuple[int, int]:
    """
        gets the coordinates of the boulders in the game map
        :param game_map: the game map
        :param symbol: the symbol of the boulder
        :return: the coordinates of the boulders
    """

    tuples = np.where(game_map == ord(symbol))
    boulders_positions = list(zip(tuples[0], tuples[1])) #converte la lista di tuple in una lista di liste
    return boulders_positions

def is_obstacle(position_element: int) -> bool:
    """
        checks if the element in the position is an obstacle
        :param position_element: the element to check
    """

    wall = "|- "
    river = "}"
    return chr(position_element) in wall or chr(position_element) == river

def get_valid_moves(game_map: np.ndarray, current_position: Tuple[int, int]) -> List[Tuple[int, int]]:
    """
        gets all the valid moves the player can make from the current position
        :param game_map: the game map as a matrix
        :param current_position: the current position of the agent
        :return: a list of valid moves  
    """

    x_limit, y_limit = game_map.shape
    valid = []
    x, y = current_position    
    
    # North valid move
    if y - 1 > 0 and not is_obstacle(game_map[x, y-1]):
        valid.append((x, y-1)) 
    # East valid move
    if x + 1 < x_limit and not is_obstacle(game_map[x+1, y]):
        valid.append((x+1, y)) 
    # South valid move
    if y + 1 < y_limit and not is_obstacle(game_map[x, y+1]):
        valid.append((x, y+1)) 
    # West valid move
    if x - 1 > 0 and not is_obstacle(game_map[x-1, y]):
        valid.append((x-1, y))
    # North-East valid move
    if x + 1 < x_limit and y - 1 > 0 and not is_obstacle(game_map[x+1, y-1]):
        valid.append((x+1, y-1))
    # South-East valid move
    if x + 1 < x_limit and y + 1 < y_limit and not is_obstacle(game_map[x+1, y+1]):
        valid.append((x+1, y+1))
    # South-West valid move
    if x - 1 > 0 and y + 1 < y_limit and not is_obstacle(game_map[x-1, y+1]):
        valid.append((x-1, y+1))
    # North-West valid move
    if x - 1 > 0 and y - 1 > 0 and not is_obstacle(game_map[x-1, y-1]):
        valid.append((x-1, y-1))

    return valid

def action_map(current_position: Tuple[int, int], new_position: Tuple[int, int]) -> Tuple[int,str]:
    """
        get the action to get to the new position from the current one
        :param new_position: the new coordinates of the agent
        :param current_position: current coordinates of the agent 
        :return: the action to get to the new position and its relative name
    """

    action = -1
    action_name = ""
    # i is raw, j is column of matrix
    i, j = current_position[0], current_position[1]
    i_new, j_new = new_position[0], new_position[1]
    action_map = {
        "N": 0,
        "E": 1,
        "S": 2,
        "W": 3,
        "NE": 4,
        "SE": 5,
        "SW": 6,
        "NW": 7
    }
    if i_new == i:
        if j_new > j:
            action = action_map["E"]
            action_name = "E"
        else: 
            action = action_map["W"]
            action_name = "W"
    elif j_new == j:
        if i_new > i:
            action = action_map["S"]
            action_name = "S"
        else: 
            action = action_map["N"]
            action_name = "N"
    elif i_new < i:
        if j_new > j:
            action = action_map["NE"]
            action_name = "NE"
        else: 
            action = action_map["NW"]
            action_name = "NW"
    elif i_new > i:
        if j_new > j:
            action = action_map["SE"]
            action_name = "SE"
        else: 
            action = action_map["SW"]
            action_name = "SW"

    return action, action_name

#TODO: exploit the action_map function
def actions_from_path(start: Tuple[int, int], path: List[Tuple[int, int]]) -> Tuple[List[int], List[str]]:
    """
        gets all the actions the player has to make to follow a path
        :param start: the starting position of the player
        :param path: the path to follow
        :return: a list of actions to perform
    """
    
    action_map = {
        "N": 0,
        "E": 1,
        "S": 2,
        "W": 3,
        "NE": 4,
        "SE": 5,
        "SW": 6,
        "NW": 7
    }
    actions = []
    action_name = []
    y_s, x_s = start
    for (y, x) in path:
        if x_s == x:
            if y_s > y:
                actions.append(action_map["N"])
                action_name.append("N")
            else: 
                actions.append(action_map["S"])
                action_name.append("S") 
        elif y_s == y:
            if x_s > x:
                actions.append(action_map["W"])
                action_name.append("W") 
            else: 
                actions.append(action_map["E"])
                action_name.append("E") 
        elif x_s < x:
            if y_s > y:
                actions.append(action_map["NE"])
                action_name.append("NE")    
            else: 
                actions.append(action_map["SE"])
                action_name.append("SE")    
        elif x_s > x:
            if y_s > y:
                actions.append(action_map["NW"])
                action_name.append("NW")    
            else: 
                actions.append(action_map["SW"])
                action_name.append("SW")    
        x_s = x
        y_s = y
        
        
    
    return actions, action_name
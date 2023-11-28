import gym
import numpy as np
import math
import time
import matplotlib.pyplot as plt
import IPython.display as display


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

def get_boulder_locations(game_map: np.ndarray, symbol : str = "`") -> List[Tuple[int, int]]:
    """
        gets the coordinates of the boulders in the game map
        :param game_map: the game map
        :param symbol: the symbol of the boulder
        :return: the coordinates of the boulders
    """

    tuples = np.where(game_map == ord(symbol))
    boulders_positions = list(zip(tuples[0], tuples[1])) #converte la lista di tuple in una lista di liste
    return boulders_positions

def get_river_locations(game_map: np.ndarray, symbol : str = "}") -> List[Tuple[int, int]]:
    """
    Returns the positions of the specified symbol in the game map.

    Parameters:
    game_map (np.ndarray): The game map represented as a numpy array.
    symbol (str): The symbol to search for in the game map. Default is "}".

    Returns:
    Tuple[int, int]: A tuple containing the row and column indices of the symbol in the game map.
    """
    tuples = np.where(game_map == ord(symbol))
    river_positions = list(zip(tuples[0], tuples[1]))
    return river_positions

def get_all_map_positions(game_map: np.ndarray) -> List[Tuple[int, int]]:
        """
        Gets all the positions in the game map
        :param game_map: the game map as a matrix
        :return: a list of all positions in the game map
        """
        positions = []
        x_limit, y_limit = game_map.shape

        for x in range(x_limit):
            for y in range(y_limit):
                positions.append((x, y))

        return positions

def is_obstacle(position_element: int, coordinates : Tuple[int,int], target_position: Tuple[int,int]) -> bool:
    """
        checks if the element in the position is an obstacle
        :param position_element: the element to check
    """

    wall = "|- "
    river = "!" ##Resolve this

    if coordinates == target_position:
        return True

    return chr(position_element) in wall or chr(position_element) == river

def get_valid_moves(game_map: np.ndarray, current_position: Tuple[int, int], target_position: Tuple[int,int]) -> List[Tuple[int, int]]:
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
    if y - 1 > 0 and not is_obstacle(game_map[x, y-1],current_position, target_position):
        valid.append((x, y-1)) 
    # East valid move
    if x + 1 < x_limit and not is_obstacle(game_map[x+1, y], current_position, target_position):
        valid.append((x+1, y)) 
    # South valid move
    if y + 1 < y_limit and not is_obstacle(game_map[x, y+1], current_position,target_position):
        valid.append((x, y+1)) 
    # West valid move
    if x - 1 > 0 and not is_obstacle(game_map[x-1, y], current_position,target_position):
        valid.append((x-1, y))
    # North-East valid move
    if x + 1 < x_limit and y - 1 > 0 and not is_obstacle(game_map[x+1, y-1], current_position,target_position): 
        valid.append((x+1, y-1))
    # South-East valid move
    if x + 1 < x_limit and y + 1 < y_limit and not is_obstacle(game_map[x+1, y+1], current_position,target_position): 
        valid.append((x+1, y+1))
    # South-West valid move
    if x - 1 > 0 and y + 1 < y_limit and not is_obstacle(game_map[x-1, y+1], current_position, target_position): 
        valid.append((x-1, y+1))
    # North-West valid move
    if x - 1 > 0 and y - 1 > 0 and not is_obstacle(game_map[x-1, y-1], current_position,target_position): 
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

def manhattan_distance(x1: int, y1: int, x2: int, y2: int):
    """
    Calculate the Manhattan distance between two positions, without considering diagonal moves.
    (4 directions)

    Parameters:
    x1 (int): The x-coordinate of the first position.
    y1 (int): The y-coordinate of the first position.
    x2 (int): The x-coordinate of the second position.
    y2 (int): The y-coordinate of the second position.

    Returns:
    int: The Manhattan distance between the two position.
    """
    return abs(x1 - x2) + abs(y1 - y2)


def chebyshev_dist(x1 : int, y1 : int, x2 : int, y2 : int): 
    """
    Calculate the Chebyshev distance between two points (x1, y1) and (x2, y2).
    Considers the diagonal moves. (8 directions)
    
    Parameters:
    x1 (int): x-coordinate of the first point.
    y1 (int): y-coordinate of the first point.
    x2 (int): x-coordinate of the second point.
    y2 (int): y-coordinate of the second point.
    
    Returns:
    int: The Chebyshev distance between the two points.
    """
    y_dist = abs(y1 - y2)
    x_dist = abs(x1 - x2)
    return max(y_dist, x_dist)


def plot_animated_sequence(env: gym.Env ,game: np.ndarray , game_map : np.ndarray, actions : List[int]):
    rewards = []
    image = plt.imshow(game[25:300, :475])
    player_positions = []
    for action in actions:
        s, r, _, _ = env.step(action)
        rewards.append(r)
        
        display.display(plt.gcf())
        display.clear_output(wait=True)
        image.set_data(s['pixel'][:, :])
        player_positions.append(get_player_location(game_map))
        time.sleep(0.5)
    print("Rewards: ")
    for r in rewards:
        print(r)
    print("Total reward: ", sum(rewards))
        
    return player_positions
import numpy as np
import math

from typing import Tuple, List

# check if an element is an obstacle(wall or river) in the game map
# @param position_element: the code of the element to check 
def is_obstacle(position_element: int) -> bool:
    obstacles = ["|- ", "}"]
    return chr(position_element) in obstacles

# get all the valid moves the player can make from the current position
# @param game_map: the game map as a matrix
# @param current_position: the current position of the player
# @return: a list of valid moves 
def get_valid_moves(game_map: np.ndarray, current_position: Tuple[int, int]) -> List[Tuple[int, int]]:
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
    # North-West valid move
    if x - 1 > 0 and y - 1 > 0 and not is_obstacle(game_map[x-1, y-1]):
        valid.append((x-1, y-1))
    # South-West valid move
    if x - 1 > 0 and y + 1 < y_limit and not is_obstacle(game_map[x-1, y+1]):
        valid.append((x-1, y+1))

    return valid
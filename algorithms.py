import numpy as np

from collections import deque
from queue import PriorityQueue
from utils import get_valid_moves, get_player_location, chebyshev_dist
from typing import Tuple, List
import matplotlib.pyplot as plt


def a_star(game : np.ndarray, game_map: np.ndarray, start: Tuple[int, int], target: Tuple[int, int], h: callable) -> List[Tuple[int, int]]:
    # initialize open and close list
    open_list = PriorityQueue()
    close_list = []
    # additional dict which maintains the nodes in the open list for an easier access and check
    support_list = {}

    starting_state_g = 0
    starting_state_h = h(start,target)
    starting_state_f = starting_state_g + starting_state_h

    open_list.put((starting_state_f, (start, starting_state_g)))
    support_list[start] = starting_state_g
    parent = {start: None}

    while not open_list.empty():
        # get the node with lowest f
        _, (current, current_cost) = open_list.get()
        # add the node to the close list
        close_list.append(current)

        if current == target:
            print("Target found!")
            print(current)
            path = build_path(parent, target, game_map, game)

            return path

    
        
        for neighbor in get_valid_moves(game_map, current, target, close_list):
            #print(get_valid_moves(game_map, current, target))
            # check if neighbor in close list, if so continue
            if neighbor in close_list:
                continue
            # compute neighbor g, h and f values
            neighbor_g = 1 + current_cost
            neighbor_h = h(neighbor, target)
            neighbor_f = neighbor_g + neighbor_h
            parent[neighbor] = current
            neighbor_entry = (neighbor_f, (neighbor, neighbor_g))
            # if neighbor in open_list
            if neighbor in support_list.keys():
                # if neighbor_g is greater or equal to the one in the open list, continue
                if neighbor_g >= support_list[neighbor]:
                    continue
            
            # add neighbor to open list and update support_list
            if neighbor == target:
                path = build_path(parent, target, game_map, game)
                return path
            open_list.put(neighbor_entry)
            support_list[neighbor] = neighbor_g

    print("Target node not found!")
    return None


def build_path(parent: dict, target: Tuple[int, int], game_map: np.ndarray, game : np.ndarray) -> List[Tuple[int, int]]:
    path = []
    while target is not None:
        path.append(target)
        agent = get_player_location(game_map)
        target = parent[target]
    path.reverse()
    return path

def print_gamestate(state):
    plt.imshow(state[100:250, 400:750]) #Immagine ristretta con range [y][x]


#Trova la minima distanza tra un punto e una serie di punti considerando movimenti diagonali
def get_min_distance_point_to_points(x, y, list_of_pairs):
    """
    Calculates the minimum distance between a point (x, y) and a list of pairs of coordinates using chebyshev distance.

    Args:
        x (int): The x-coordinate of the point.
        y (int): The y-coordinate of the point.
        list_of_pairs (list): A list of pairs of coordinates.

    Returns:
        tuple: A tuple containing the coordinates of the pair with the minimum distance and the minimum distance itself.
    """
    min_dist = 999999999
    for i in list_of_pairs:
        dist = chebyshev_dist(x, y, i[0], i[1])
        if dist < min_dist:
            min_dist = dist
            coordinates = [i[0], i[1]]
    return coordinates, min_dist

def get_optimal_distance_point_to_point(start: Tuple[int, int], target : Tuple[int,int]) -> int:
    """
    Calculate the optimal distance between two points.

    Args:
        start (Tuple[int, int]): The starting point coordinates.
        target (Tuple[int, int]): The target point coordinates.

    Returns:
        int: The optimal distance between the two points.
    """
    targetX = target[0]
    targetY = target[1]
    dist_point_to_point = chebyshev_dist(start[0],start[1], targetX, targetY)
    
    dist = dist_point_to_point #-1

    return dist



def get_best_global_distance(start: Tuple[int, int], boulders: List[Tuple[int,int]], river_positions : List[Tuple[int,int]]) -> Tuple[int, int]:
    """
    Calculates the best global distance between the start point, boulders, and river positions.

    Args:
        start (Tuple[int, int]): The starting point coordinates.
        boulders (List[Tuple[int,int]]): List of boulder coordinates.
        river_positions (List[Tuple[int,int]]): List of river position coordinates.

    Returns:
        Tuple[int, int]: The coordinates of the boulder with the best global distance.
    """
   
    distances = []
    
    for boulder in boulders:
        x = boulder[0]
        y = boulder[1]

        dist_player_boulder = get_optimal_distance_point_to_point(start, boulder)
        
        dist_boulder_river = get_min_distance_point_to_points(boulder[0],boulder[1], river_positions)
        dist = dist_player_boulder + dist_boulder_river[1] #position 1 is just the value
        distances.append((x, y, dist))

    min_distance = min(distances, key=lambda x: x[2])
    return min_distance[0], min_distance[1]

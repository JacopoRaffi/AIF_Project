import numpy as np

from collections import deque
from queue import PriorityQueue
from utils import *
from typing import Tuple, List
import matplotlib.pyplot as plt



def print_gamestate(state):
    """
    Prints the game state.

    Parameters:
    state (numpy.ndarray): The game state array.

    Returns:
    None
    """
    plt.imshow(state[100:250, 400:750]) 

def a_star(game : np.ndarray, game_map: np.ndarray, start: Tuple[int, int], target: Tuple[int, int], h: callable) -> List[Tuple[int, int]]:
    """
    A* algorithm implementation to find the shortest path from the start position to the target position on a game map.

    Parameters:
    - game: numpy.ndarray - The game state.
    - game_map: numpy.ndarray - The map of the game.
    - start: Tuple[int, int] - The starting position.
    - target: Tuple[int, int] - The target position.
    - h: callable - The heuristic function to estimate the distance between two positions.

    Returns:
    - List[Tuple[int, int]] - The list of positions representing the shortest path from the start to the target.
    """
    count = 0 #Variable needed to keep track of the order of insertion in the priority queue, if two items have the same f score the one with the lower count is inserted first
    open_set = PriorityQueue()
    open_set.put((0, count, start)) #f score, count, position
    came_from = {} #Tracks from which node we reached a node
    g_scores = {} #Dictionary with the g score of each node
    f_scores = {} #Dictionary with the f score of each node

    map_positions = get_all_map_positions(game_map) #Get all the positions x,y in the map

    #Set the g and f score of all the nodes to infinity in the dictionary
    for position in map_positions:
        g_scores[position] = float("inf")
        f_scores[position] = float("inf")

    #Scores of the start node
    g_start = 0 #g score of the start node
    f_start= h(start, target) #f score of the start node
    
    #Dictionary with the g score of each node
    g_scores[start] = g_start #Insert the g score of the start node in the dictionary
    f_scores[start] = f_start #Insert the f score of the start node in the dictionary
 
    open_set_hash = {start} #Track all the items in the priority queue, we need this because we can't check if an item directly in the priority queue

    while not open_set.empty():
        current = open_set.get()[2] #Get the position of the current node, get the item with the lowest f score
        open_set_hash.remove(current) #Remove the current node from the open set

        if current == target:
            print("Target found!")
            path = reconstruct_path(came_from, current) #Reconstruct the path from the start node to the target node
            return path
        
        for neighbour in get_valid_moves(game_map, current, target): #Neighbours of the current node
            
            temp_g_score = g_scores[current] + 1 #g score of the neighbour calulated as the g score of the current node + 1
            
            if temp_g_score < g_scores[neighbour]: #if we found a better way to reach this neighbour update the g score
                came_from[neighbour] = current #Update the node from which we reached the neighbour
                g_scores[neighbour] = temp_g_score #Update the g score of the neighbour
                f_scores[neighbour] = temp_g_score + h(neighbour, target) #Update the f score of the neighbour

                if neighbour not in open_set_hash:
                    count += 1
                    open_set.put((f_scores[neighbour], count, neighbour)) #Add the neighbour to the open set because it is the best path to reach the target
                    open_set_hash.add(neighbour)
    return None

def reconstruct_path(came_from, current):
    """
    Reconstructs the path from the start node to the current node using the 'came_from' dictionary.

    Args:
        came_from (dict): A dictionary that maps each node to its previous node in the path.
        current: The current node.

    Returns:
        list: The reconstructed path from the start node to the current node.
    """
    path = []
    #Follows the path from the end to the starta
    while current in came_from:
        current = came_from[current] 
        path.append(current)
    return path[::-1] #reverse the path
        

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
        dist = get_optimal_distance_point_to_point((x, y), (i[0], i[1]))
        if dist < min_dist or (dist == min_dist and x == i[0]): #if the distance is the same prefers the one with the same column index
            min_dist = dist
            coordinates = [i[0], i[1]]
    return coordinates[0], coordinates[1]

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

    print(distances)
    min_distance = min(distances, key=lambda x: x[2])
    return min_distance[0], min_distance[1]

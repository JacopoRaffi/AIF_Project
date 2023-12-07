import numpy as np

from collections import deque
from queue import PriorityQueue
from utils import *
from typing import Tuple, List
import matplotlib.pyplot as plt
from logic import position_for_boulder_push, push_boulder_path



def print_gamestate(state):
    """
    Prints the game state.

    Parameters:
    state (numpy.ndarray): The game state array.

    Returns:
    None
    """
    plt.imshow(state[100:250, 400:750]) 

def a_star(game_map: np.ndarray, start: Tuple[int, int], target: Tuple[int, int], hasBoulder:bool, h: callable) -> List[Tuple[int, int]]:
    """
    A* algorithm implementation to find the shortest path from the start position to the target position on a game map.

    Parameters:
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
            path = reconstruct_path(came_from, current) #Reconstruct the path from the start node to the target node
            return path
        
        for neighbour in get_valid_moves(game_map, current, target,hasBoulder): #Neighbours of the current node

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
    path = [current]
    #Follows the path from the end to the starta
    while current in came_from:
        current = came_from[current] 
        path.append(current)
    return path[::-1] #reverse the path
        
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
    coordinates = []
    min_dist = float("inf")
    for i in list_of_pairs:
        dist = get_optimal_distance_point_to_point((x, y), (i[0], i[1]))
        if dist < min_dist or (dist == min_dist and x == i[0]): #if the distance is the same prefers the one with the same column index
            min_dist = dist
            coordinates = [i[0], i[1]]
    return tuple(coordinates), min_dist

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
        distances.append((x, y, dist, dist_boulder_river[1]))

    min_distance = min(distances, key=lambda x: (x[2], x[3]))
    return min_distance[0], min_distance[1]
'''
def push_one_boulder_into_river_OLD(state, env : gym.Env, target=None): 
    """
    Pushes one boulder into the river in the game environment.

    Args:
        state (dict): The current state of the game.
        env (gym.Env): The game environment.
        target (tuple, optional): The target position for pushing the boulder. Defaults to None.
        When specificed the algorithm will push the boulder into the specified river position.

    Returns:
        tuple: The new target position for pushing the next boulder.
    """
    
    game_map = state['chars']
    game = state['pixel']

    start = get_player_location(game_map)
    boulders = get_boulder_locations(game_map)
    river_positions = get_river_locations(game_map)

    #If there is no target means that is the first boulder pushed into the river
    #then proceed to find the best boulder to push into the river within one of the river positions
    if target is None:  
        coordinates_min_boulder = get_best_global_distance(start, boulders, river_positions)
        temp = get_min_distance_point_to_points(coordinates_min_boulder[0],coordinates_min_boulder[1], river_positions)
        final_position = tuple(temp[0])
    else: #We specific next river position in which we have to drop the boulder
        coordinates_min_boulder = get_best_global_distance(start, boulders, [target])
        final_position = target


    #Calculating the path from the boulder to the river shortest distance
    path_boulder_river = a_star(game_map, coordinates_min_boulder,final_position, get_optimal_distance_point_to_point)
    path_boulder_river.append(final_position) 

    #Calculating the position in which the agent have to be in order to push correctly the boulder into the river
    pushing_position = position_for_boulder_push(coordinates_min_boulder, path_boulder_river[1])[1]
    

    #Calculating the path from the player to the pushing position
    path_player_to_pushing_position = a_star(game_map, start,  pushing_position, get_optimal_distance_point_to_point)

    #Correcting the path from the player to the pushing position
    agent_actions,path_player_to_river = push_boulder_path(path_boulder_river)


    if(path_player_to_pushing_position is not None):
        if(path_player_to_river is not None):
            agent_full_path = path_player_to_pushing_position + path_player_to_river
        else:
            agent_full_path = path_player_to_pushing_position
    else:
        if(path_player_to_river is not None):
            agent_full_path = path_player_to_river
        else:
            agent_full_path = None

    actions, names = actions_from_path(start, agent_full_path) 

    player_pos = plot_animated_sequence(env,game, game_map,actions[1:]) #Remove first action because it is the start position

    #Computes and return the new target which is the next river position
    new_target = final_position = (final_position[0], final_position[1])
    return new_target
'''

def push_one_boulder_into_river(state, env : gym.Env, target=None): 
    """
    Pushes one boulder into the river in the game environment.

    Args:
        state (dict): The current state of the game.
        env (gym.Env): The game environment.
        target (tuple, optional): The target position for pushing the boulder. Defaults to None.
        When specificed the algorithm will push the boulder into the specified river position.

    Returns:
        tuple: The new target position for pushing the next boulder.
    """
    
    game_map = state['chars']
    game = state['pixel']

    start = get_player_location(game_map)
    boulders = get_boulder_locations(game_map)
    river_positions = get_river_locations(game_map)

    #If there is no target means that is the first boulder pushed into the river
    #then proceed to find the best boulder to push into the river within one of the river positions
    if target is None:  
        coordinates_min_boulder = get_best_global_distance(start, boulders, river_positions)
        temp = get_min_distance_point_to_points(coordinates_min_boulder[0],coordinates_min_boulder[1], river_positions)
        final_position = tuple(temp[0])
    else: #We specific next river position in which we have to drop the boulder
        coordinates_min_boulder = get_best_global_distance(start, boulders, [target])
        final_position = target

    #Calculating the path from the boulder to the river shortest distance
    path_boulder_river = a_star(game_map, coordinates_min_boulder,final_position, get_optimal_distance_point_to_point)
    #path_boulder_river.append(final_position) 

    #Calculating the position in which the agent have to be in order to push correctly the boulder into the river
    pushing_position = position_for_boulder_push(coordinates_min_boulder, path_boulder_river[1])[1]
    

    #Calculating the path from the player to the pushing position
    path_player_to_pushing_position = a_star(game_map, start,  pushing_position, get_optimal_distance_point_to_point)

    #Correcting the path from the player to the pushing position
    agent_actions,path_player_to_river = push_boulder_path(path_boulder_river)
    path_player_to_river = path_player_to_river[1:] #Remove the first element because the agent is already in the pushing position


    if(path_player_to_pushing_position is not None):
        if(path_player_to_river is not None):
            agent_full_path = path_player_to_pushing_position + path_player_to_river
        else:
            agent_full_path = path_player_to_pushing_position
    else:
        if(path_player_to_river is not None):
            agent_full_path = path_player_to_river
        else:
            agent_full_path = None

    online_a_star(start, agent_full_path, env, game_map, coordinates_min_boulder) #Start to walk and recompute the path if needed
    
def check_better_path(new_map, current_target, actual_target=None):
    """
        checks if there is a better path to follow after a change of state of the map
        :param new_map: the new state after the agent's step
        :param river: the positions of the river water blocks
        :param actual_target: the actual target of the agent for the first push of the boulder (-1,-1) means there isn't an actual_target
        :return: the new path to follow
    """

    if actual_target is not None:
        if not is_obstacle(new_map[actual_target], get_player_location(new_map), actual_target):
            new_path = a_star(new_map, get_player_location(new_map), actual_target, False, get_optimal_distance_point_to_point)
            return new_path
    
    new_path = a_star(new_map, get_player_location(new_map), current_target, False, get_optimal_distance_point_to_point)
    new_path = new_path[:-1]
    return new_path

def push_new_boulder(old_map, new_map, agent_pos, river, current_boulder, boulder_symbol='`'):
    """
        checks if there it is more convinient to push a new boulder
        :param old_map: the previous state of the map
        :param new_map: the new state after the agent's step
        :param agent_pos: the position of the agent
        :param river: the positions of the river water blocks
        :return: the new path to follow (iff there are changes on the state) and the actual target for the first push of the boulder and
                the best new boulder to push
                path, None means that the final element of the path is the actual target 
                None,None means do nothing new
    """

    old_pos = get_boulder_locations(old_map, boulder_symbol)
    new_pos = get_boulder_locations(new_map, boulder_symbol)

    if old_pos != new_pos: #if there is at least one new boulder seen by the agent after the step 
        new_boulder = get_best_global_distance(agent_pos, new_pos, river)
       
        if new_boulder == current_boulder: #if the new boulder is the same as the current one
            return None, None, current_boulder

        temp = get_min_distance_point_to_points(new_boulder[0], new_boulder[1], river)
        river_target = tuple(temp[0])
        boulder_to_river = a_star(new_map, new_boulder, river_target, True, get_optimal_distance_point_to_point)

        _,agent_first_push = position_for_boulder_push(new_boulder, boulder_to_river[1]) # get the first position the agent needs to be to push the boulder

        if not is_obstacle(new_map[agent_first_push], agent_pos, agent_first_push): #if the agent_first_push is not an obstacle
            agent_to_boulder = a_star(new_map, agent_pos,  agent_first_push, False, get_optimal_distance_point_to_point) #get the new path to follow 
            actual_target = agent_first_push

        else:
            agent_to_boulder = a_star(new_map, agent_pos, new_boulder, False, get_optimal_distance_point_to_point) #get the new path to follow
            agent_to_boulder = agent_to_boulder[:-1]
            actual_target = None

        return agent_to_boulder, actual_target, new_boulder
    
    else:
        return None, None, current_boulder
    
def online_a_star(start: Tuple[int, int], path : [List[Tuple[int,int]]], env : gym.Env, game_map : np.ndarray, current_boulder : Tuple[int,int], boulder_symbol='`'):

    old_map = new_map = game_map #Initialize the old and new map with the current game map

    while len(path) > 0:
        old_map = new_map #Map at timestep t-1
        actions, names= actions_from_path(start, path) #Get the actions to follow the path
        observation, reward, done, info = env.step(actions[0]) #Execute the first action
        new_map = observation['chars'] #Update the new map after the step

        if(are_less_black_blocks(new_map, old_map)): #if there are less black blocks than before
            newpath, true_pushing_position = push_new_boulder(old_map, new_map, start, get_river_locations(new_map), current_boulder)

            if(newpath == None): #The boulder to push is the same as before
                path_temp = check_better_path(new_map, get_river_locations(new_map), true_pushing_position, actual_target=true_pushing_position)

                if(len(path) > len(path_temp)): #i found a shorter path
                    path = path_temp 
                else:
                    path = path[1:]
                    continue
            else: #The boulder to push has changed
                path = newpath
                continue
        
        path = path[1:] #Remove the first action because it has already been executed
        '''
        if(player è bloccato) 
            avoid_obstacle()
            path = a_star(…)
        '''
        '''
        if(vedo nuove caselle nere) 
            new_path, true_pushing_position = push_new_boulder(…) #new path ha il boulder da pushare
            
            if(new path == None) # Significa che il masso è lo stesso di prima
                path_temp = check_better_path( … )
                
                if(path.length( ) > path_temp.length( ) #Ho trovato un path più breve
                    path = path_temp
                else
                    path = path[1:] #Tolgo la prima azione che ho già fatto e continuo
                    continue #segue il path originale
            else #Significa che il masso da spostare è cambiato
                path = new_path
                continue
	
	
        path = path[1:] #Tolgo la prima azione che ho già fatto e continuo
        '''
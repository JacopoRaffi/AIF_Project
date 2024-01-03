import numpy as np

from collections import deque
from queue import PriorityQueue
from utils import *
from typing import Tuple, List
import matplotlib.pyplot as plt
from logic import *
import IPython.display as display


def print_gamestate(state):
    """
        Prints the game state.

        Parameters:
        - state (numpy.ndarray): The game state array.

        Returns:
        - None
    """

    plt.imshow(state[100:250, 400:750]) 

def get_optimal_distance_point_to_point(start: Tuple[int, int], target: Tuple[int,int]) -> int:
    """
        Calculates the optimal distance between two points.

        Parameters:
        - start (Tuple[int, int]): The starting point coordinates.
        - target (Tuple[int, int]): The target point coordinates.

        Returns:
        - dist (int): The optimal distance between the two points.
    """

    targetX = target[0]
    targetY = target[1]
    dist_point_to_point = chebyshev_dist(start[0],start[1], targetX, targetY)
    
    dist = dist_point_to_point #-1

    return dist

def a_star(game_map: np.ndarray, start: Tuple[int, int], target: Tuple[int, int], hasBoulder: bool, boulder_is_obstacle=False, h=get_optimal_distance_point_to_point) -> List[Tuple[int, int]]:
    """
        A* algorithm implementation to find the shortest path from the start position to the target position on a game map.

        Parameters:
        - game_map (numpy.ndarray): The map of the game.
        - start (Tuple[int, int]): The starting position.
        - target (Tuple[int, int]): The target position.
        - hasBoulder (bool): True if the player is carrying a boulder, False otherwise.
        - boulder_is_obstacle (bool): True if the boulder is considered as an obstacle, False otherwise.
        - h (callable): The heuristic function to estimate the distance between two positions.

        Returns:
        - List[Tuple[int, int]]: The list of positions representing the shortest path from the start to the target.
    """

    count = 0 # Variable needed to keep track of the order of insertion in the priority queue, if two items have the same f score the one with the lower count is inserted first
    open_set = PriorityQueue()
    open_set.put((0, count, start)) # f score, count, position
    came_from = {} # Tracks from which node we reached a node
    g_scores = {} # Dictionary with the g score of each node
    f_scores = {} # Dictionary with the f score of each node

    map_positions = get_all_map_positions(game_map) # Get all the positions x,y in the map

    # Set the g and f score of all the nodes to infinity in the dictionary
    for position in map_positions:
        g_scores[position] = float("inf")
        f_scores[position] = float("inf")

    # Scores of the start node
    g_start = 0 # g score of the start node
    f_start= h(start, target) # f score of the start node
    
    # Dictionary with the g score of each node
    g_scores[start] = g_start # Insert the g score of the start node in the dictionary
    f_scores[start] = f_start # Insert the f score of the start node in the dictionary
 
    open_set_hash = {start} # Track all the items in the priority queue, we need this because we can't check if an item directly in the priority queue

    while not open_set.empty():
        current = open_set.get()[2] # Get the position of the current node, get the item with the lowest f score
        open_set_hash.remove(current) # Remove the current node from the open set

        if current == target:
            path = reconstruct_path(came_from, current) # Reconstruct the path from the start node to the target node
            return path
        
        for neighbour in get_valid_moves(game_map, current, target,hasBoulder, boulder_is_obstacle): # Neighbours of the current node

            temp_g_score = g_scores[current] + 1 # g score of the neighbour calulated as the g score of the current node + 1
            
            if temp_g_score < g_scores[neighbour]: # If we found a better way to reach this neighbour update the g score
                came_from[neighbour] = current # Update the node from which we reached the neighbour
                g_scores[neighbour] = temp_g_score # Update the g score of the neighbour
                f_scores[neighbour] = temp_g_score + h(neighbour, target) # Update the f score of the neighbour

                if neighbour not in open_set_hash:
                    count += 1
                    open_set.put((f_scores[neighbour], count, neighbour)) # Add the neighbour to the open set because it is the best path to reach the target
                    open_set_hash.add(neighbour)
            
    return None

def reconstruct_path(came_from, current):
    """
        Reconstructs the path from the start node to the current node using the 'came_from' dictionary.

        Parameters:
        - came_from (dict): A dictionary that maps each node to its previous node in the path.
        - current: The current node.

        Returns:
        - List[Tuple[int, int]]: The reconstructed path from the start node to the current node.
    """

    path = [current]
    # Follows the path from the end to the starta
    while current in came_from:
        current = came_from[current] 
        path.append(current)
    return path[::-1] # Reverse the path
        
def get_min_distance_point_to_points(x, y, list_of_pairs):
    """
        Calculates the minimum distance between a point (x, y) and a list of pairs of coordinates using chebyshev distance.

        Parameters:
        - x (int): The x-coordinate of the point.
        - y (int): The y-coordinate of the point.
        - list_of_pairs (list): A list of pairs of coordinates.

        Returns:
        - tuple: A tuple containing the coordinates of the pair with the minimum distance and the minimum distance itself.
    """

    coordinates = []
    min_dist = float("inf")
    for i in list_of_pairs:
        dist = get_optimal_distance_point_to_point((x, y), (i[0], i[1]))
        if dist < min_dist or (dist == min_dist and x == i[0]): # If the distance is the same prefers the one with the same raw index
            min_dist = dist
            coordinates = [i[0], i[1]]
    return tuple(coordinates), min_dist

def get_best_global_distance(start: Tuple[int, int], boulders: List[Tuple[int,int]], river_positions: List[Tuple[int,int]]) -> Tuple[int, int]:
    """
        Calculates the best global distance between the start point, boulders, and river positions.

        Parameters:
        - start (Tuple[int, int]): The starting point coordinates.
        - boulders (List[Tuple[int,int]]): List of boulder coordinates.
        - river_positions (List[Tuple[int,int]]): List of river position coordinates.

        Returns:
        - Tuple[int, int]: The coordinates of the boulder with the best global distance.
    """
   
    distances = []
    
    for boulder in boulders:
        x = boulder[0]
        y = boulder[1]

        dist_player_boulder = get_optimal_distance_point_to_point(start, boulder)
        
        dist_boulder_river = get_min_distance_point_to_points(boulder[0],boulder[1], river_positions)
        dist = dist_player_boulder + dist_boulder_river[1] # Position 1 is just the value
        distances.append((x, y, dist, dist_boulder_river[1]))

    
    if len(distances) == 0:
        return -1, -1
    min_distance = min(distances, key=lambda x: (x[2], x[3]))
    return min_distance[0], min_distance[1]

def push_one_boulder_into_river(state, env: gym.Env, black_list_boulder, target=None, plot=True): 
    """
        Pushes one boulder into the river in the game environment.

        Parameters:
        - state (dict): The current state of the game.
        - env (gym.Env): The game environment.
        - black_list_boulder (list): List of boulder coordinates that we can't push.
        - target (tuple, optional): The target position for pushing the boulder. Defaults to None.
                                    When specificed the algorithm will push the boulder into the specified river position.
        - plot (bool, optional): True if the animated sequence of the agent has to be plotted, False otherwise. Defaults to True.
    
        Returns:
        - tuple: The new target position for pushing the next boulder.
    """
    
    game_map = state['chars']
    game = state['pixel']

    
    start = get_player_location(game_map)

    if start is None:
        return None, None, None

    boulders = get_boulder_locations(game_map, black_list_boulder)
    
    river_positions = find_river(env, game_map, black_list_boulder)
    

    # If there is no target means that is the first boulder pushed into the river
    # then proceed to find the best boulder to push into the river within one of the river positions
    if target is None:  
        coordinates_min_boulder = get_best_global_distance(start, boulders, river_positions)
        temp = get_min_distance_point_to_points(coordinates_min_boulder[0],coordinates_min_boulder[1], river_positions)
        final_position = tuple(temp[0])
    else: # We specific next river position in which we have to drop the boulder
        coordinates_min_boulder = get_best_global_distance(start, boulders, [target])
        final_position = target

    hasBoulder = True # The river is not considered as an obstacle

    # Calculating the path from the boulder to the river shortest distance
    path_boulder_river = a_star(game_map, coordinates_min_boulder,final_position, hasBoulder, False, get_optimal_distance_point_to_point)

    # Calculating the position in which the agent have to be in order to push correctly the boulder into the river
    pushing_position = position_for_boulder_push(coordinates_min_boulder, path_boulder_river[1])[1]
    nearest_pushing_position = pushing_position
    if game_map[pushing_position] == ord(" "): # The target is an unseen block
        nearest_pushing_position = get_neighbour_pushing_position(game_map, pushing_position, coordinates_min_boulder) # !!!Nearest position to the boulder pushing pos
        hasBoulder = False # The river is considered as an obstacle
        path_player_to_pushing_position = a_star(game_map, start, nearest_pushing_position, hasBoulder, True, get_optimal_distance_point_to_point)

    else:
        hasBoulder = False # The river is considered as an obstacle
        # Calculating the path from the player to the pushing position
        path_player_to_pushing_position = a_star(game_map, start,  pushing_position,hasBoulder,False, get_optimal_distance_point_to_point)


    #Correcting the path from the player to the pushing position
    agent_actions,path_player_to_river = push_boulder_path(path_boulder_river)
    if pushing_position == nearest_pushing_position:
        path_player_to_river = path_player_to_river[1:] # Remove the first element because the agent is already in the pushing position

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
    
    return online_a_star(start, agent_full_path, env, game_map,game ,pushing_position, nearest_pushing_position, coordinates_min_boulder, path_boulder_river[-1], black_list_boulder, plot) #Start to walk and recompute the path if needed
    
def check_better_path(new_map, current_target, actual_target=None):
    """
        Checks if there is a better path to follow after a change of state of the map

        Parameters:
        - new_map (numpy.ndarray): The new state after the agent's step.
        - current_target (tuple): The current target of the agent.
        - actual_target (tuple): The actual target of the agent for the first push of the boulder (-1,-1) means there isn't an actual_target.
        
        Return:
        - new_path (list): The new path to follow.
    """

    if actual_target is not None:
        if not is_obstacle(new_map[actual_target], get_player_location(new_map), actual_target):
            new_path = a_star(new_map, get_player_location(new_map), actual_target, False, False, get_optimal_distance_point_to_point)
            return new_path
    
    new_path = a_star(new_map, get_player_location(new_map), current_target, False, True, get_optimal_distance_point_to_point)
    return new_path

def check_boulder_to_river(new_map, current_boulder):
    """
        Checks if there is a better path to follow (for boulder) after a change of state of the map

        Parameters:
        - new_map (numpy.ndarray): The new state after the agent's step.
        - current_boulder (tuple): The current boulder to push.

        Return:
        - agent_new_path (list): The new path to follow.
        - new_first_pushing_position (tuple): The new first pushing position.
        - new_river_target (tuple): The new river target.
        - new_path (list): The new path to follow (boulder to river).
    """

    river_positions = get_river_locations(new_map)

    new_river_target, _ = get_min_distance_point_to_points(current_boulder[0], current_boulder[1], river_positions)

    new_path = a_star(new_map, current_boulder, new_river_target, True, get_optimal_distance_point_to_point) # Compute new boulder path
    _,new_first_pushing_position = position_for_boulder_push(current_boulder, new_path[1]) # Get the first position the agent needs to be to push the boulder
    _,agent_new_path = push_boulder_path(new_path) # Compute new agent path needs to follow to push the boulder into the river
    
    return agent_new_path, new_first_pushing_position, new_river_target, new_path[0]

def push_new_boulder(old_map, new_map, agent_pos, river, nearest_first_pushing_pos, current_boulder, black_list_boulder, boulder_symbol='`'):
    """
        Checks if there it is more convinient to push a new boulder

        Parameters:
        - old_map (numpy.ndarray): The previous state of the map.
        - new_map (numpy.ndarray): The new state after the agent's step.
        - agent_pos (tuple): The position of the agent.
        - river (list): The positions of the river water blocks.
        - nearest_first_pushing_pos (tuple): The nearest first pushing position.
        - current_boulder (tuple): The current boulder to push.
        - black_list_boulder (list): List of boulder coordinates that we can't push.
        - boulder_symbol (str): The boulder symbol.

        Return:
        - list: The new path to follow.
        - tuple: The new boulder to push.
        - tuple: The first pushing position.
        - tuple: The nearest pushing position.
    """

    old_pos = get_boulder_locations(old_map, black_list_boulder,boulder_symbol)
    new_pos = get_boulder_locations(new_map, black_list_boulder,boulder_symbol)
    

    if len(old_pos) != len(new_pos): # If there is at least one new boulder seen by the agent after the step 
        new_boulder = get_best_global_distance(agent_pos, new_pos, river)
        
        if(new_boulder == (-1,-1)): # If there are no more boulders to push
            return None, None, None, None
        temp = get_min_distance_point_to_points(new_boulder[0], new_boulder[1], river)
        river_target = tuple(temp[0])
        boulder_to_river = a_star(new_map, new_boulder, river_target, True, get_optimal_distance_point_to_point)

        _,agent_first_push = position_for_boulder_push(new_boulder, boulder_to_river[1]) # Get the first position the agent needs to be to push the boulder
        nearest_pushing_position = agent_first_push
        if not is_obstacle(new_map[agent_first_push], agent_pos, agent_first_push): # If the agent_first_push is not an obstacle
            agent_to_boulder = a_star(new_map, agent_pos,  agent_first_push, False, get_optimal_distance_point_to_point) # Get the new path to follow 
            first_pushing_position = None

        else:
            nearest_pushing_position = get_neighbour_pushing_position(new_map, agent_first_push, new_boulder)
            agent_to_boulder = a_star(new_map, agent_pos, nearest_pushing_position, False, get_optimal_distance_point_to_point) # Get the new path to follow
            first_pushing_position = agent_first_push

        return agent_to_boulder, new_boulder, first_pushing_position, nearest_pushing_position
    
    else:
        return None, current_boulder, None, nearest_first_pushing_pos
    
def online_a_star(start: Tuple[int, int], path: [List[Tuple[int,int]]], env: gym.Env, game_map: np.ndarray, game: np.ndarray, first_pushing_position: Tuple[int,int], nearest_pushing_position: Tuple[int,int], current_boulder: Tuple[int,int], river_target, black_list_boulder, plot=True):
    """
        Executes the online A* algorithm.

        Parameters:
        - start (Tuple[int, int]): The starting position.
        - path (List[Tuple[int,int]]): The path to follow.
        - env (gym.Env): The game environment.
        - game_map (numpy.ndarray): The map of the game.
        - game (numpy.ndarray): The game state.
        - first_pushing_position (Tuple[int,int]): The first pushing position.
        - nearest_pushing_position (Tuple[int,int]): The nearest pushing position.
        - current_boulder (Tuple[int,int]): The current boulder to push.
        - river_target (Tuple[int,int]): The river target.
        - black_list_boulder (list): List of boulder coordinates that we can't push.
        - plot (bool, optional): True if the animated sequence of the agent has to be plotted, False otherwise. Defaults to True.

        Returns:
        - numpy.ndarray: The new state of the game.
        - Tuple[int,int]: The new river target.
        - int: The length of the path.
    """
    old_map = new_map = game_map.copy() # Initialize the old and new map with the current game map 
    
    if plot:
        image = plt.imshow(game[25:300, :475]) # Plotting the initial image


    current_river_target = river_target
    final_path = [] # For debugging and evaluation
    path_length = len(path)
    while path_length > 1:
        final_path.append(path.pop(0)) # Remove the first action because it has already been executed
        path_length = len(path) # Update the length of the path
        old_map = new_map.copy() # Map at timestep t-1
        actions, names = actions_from_path(start, path) # Get the actions to follow the path
        observation, reward, done, info = env.step(actions[0]) # Execute the first action

        if plot:
            plot_anim_seq_online_a_star(observation, image) # Plots the animated sequence of the agent

        if(len(path) == 1): # Finish the execution without computing a path with a new boulder
            return observation, current_river_target, len(final_path)

        new_map = observation['chars'] # Update the new map after the step
        start = get_player_location(new_map) # Update the start position for the next iteration
        prev_position = get_player_location(old_map) # Update the prev player position

        if is_player_same_position(start, prev_position):
            state, result, river_target = avoid_obstacle(game_map, start, actions[0], env)
            if result == 0:
                return state, None, None
            elif result == 1:
                return state, river_target, len(final_path)
            
            elif result == 2:
                black_list_boulder.append(current_boulder)
                return state, None, black_list_boulder

        if(are_less_black_blocks(new_map, old_map)): # If there are less black blocks than before
            newpath, current_boulder, true_pushing_position, nearest_pushing_position = push_new_boulder(old_map, new_map, start, get_river_locations(new_map), nearest_pushing_position, current_boulder, black_list_boulder)
            if current_boulder is None:
                return observation, None, None   
            # Update path boulder to river iff near the boulder
            path_temp2, first2, new_river_target, tmp_boulder = check_boulder_to_river(new_map, current_boulder)

            if(newpath == None): # The boulder to push is the same as before
                path_temp = check_better_path(new_map, nearest_pushing_position, actual_target=first_pushing_position)
                
                if path_temp is None:
                    return observation, None, None

                if path_temp[-1] == path_temp2[0]:
                    final_path_temp = path_temp[:-1] + path_temp2 # Concatenate the two new path
                else:
                    final_path_temp = path_temp + path_temp2

                if(len(path) > len(final_path_temp)): # I found a shorter path
                    path = final_path_temp 
                    first_pushing_position = first2
                    current_river_target = new_river_target
                    current_boulder = tmp_boulder
            else: # The boulder to push has changed
                if newpath[-1] == path_temp2[0]:
                    final_path_temp = newpath[:-1] + path_temp2
                else:
                    final_path_temp = newpath + path_temp2

                path = final_path_temp
                first_pushing_position = true_pushing_position # Update the first pushing position
                current_river_target = new_river_target # Update the current river target

    return observation, current_river_target, len(final_path) # Return just for test purposes (it will be removed)

def get_neighbour_pushing_position(game_map: np.ndarray, pushing_position: Tuple[int, int], boulder_position: Tuple[int, int]):
    """
        Returns the neighbour of the pushing position.

        Parameters:
        - game_map (np.ndarray): The game map.
        - pushing_position (Tuple[int, int]): The pushing position.
        - boulder_position (Tuple[int, int]): The boulder position.

        Returns:
        - Tuple[int, int]: The neighbour of the pushing position.
    """

    neighbours = []
    x, y = pushing_position

    # Moves: up, down, left, right, up-left, down-right, up-right, down-left 
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1), (0, 2), (0, -2), (2, 0), (-2, 0), (2,2), (-2,-2), (2,-2), (-2,2)]

    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if 0 <= nx < game_map.shape[0] and 0 <= ny < game_map.shape[1]:
            if not is_obstacle(game_map[nx, ny], pushing_position, (nx, ny)) and not (nx, ny) == boulder_position:
                neighbours.append((nx, ny))
    
    return neighbours[0]
    
def find_exit(env: gym.Env , game_map: np.ndarray):
    """
        Finds the exit position in the game map.

        Parameters:
        - env (gym.Env): The game environment.
        - game_map (np.ndarray): The game map.

        Returns:
        - Tuple[float, int]: The reward and the number of steps needed to reach the exit.
    """
    
    player_pos = get_player_location(game_map)
    exit_pos = find_stairs(env, game_map)
    rewards = [0.0]
    steps = 0

    if exit_pos is None:
        return 0.0, 0 

    path_to_exit = a_star(game_map, player_pos, exit_pos, False, False, get_optimal_distance_point_to_point)
    actions_to_exit,names = actions_from_path(player_pos, path_to_exit[1:])

    for action in actions_to_exit:
        s, r, _, _ = env.step(action)
        steps = steps + 1

        if get_player_location(s['chars']) == None:
            rewards.append(r)
            break
        else:
            rewards.append(r)
        
    if rewards[-1] is not None:
        return rewards[-1], steps
    else:
        return 0.0, steps

import gym
import minihack
import numpy as np
import math
import time
import matplotlib.pyplot as plt
import IPython.display as display
from typing import List
import os
from datetime import datetime
from typing import Tuple, List
from minihack import MiniHackNavigation, LevelGenerator


def get_player_location(game_map: np.ndarray, symbol: str = "@") -> Tuple[int, int]:
    """
        gets the coordinates of the player in the game map
        :param game_map: the game map
        :param symbol: the symbol of the agent
        :return: the coordinates of the agent
    """
    
    x, y = np.where(game_map == ord(symbol))
    if len(x) == 0 or len(y) == 0:
        return None
    return (x[0], y[0])

def get_exit_location(game_map: np.ndarray, symbol: str = ">") -> Tuple[int, int]:
    """
        gets the coordinates of the exit in the game map
        :param game_map: the game map
        :param symbol: the symbol of the exit
        :return: the coordinates of the exit
    """

    return get_player_location(game_map, symbol)

def get_boulder_locations(game_map: np.ndarray, black_list_boulder, symbol: str = "`") -> List[Tuple[int, int]]:
    """
        gets the coordinates of the boulders in the game map
        :param game_map: the game map
        :param black_list_boulder: the list of boulders to be removed
        :param symbol: the symbol of the boulder
        :return: the coordinates of the boulders
    """
    tuples = np.where(game_map == ord(symbol)) 
    
    boulders_positions = list(zip(tuples[0], tuples[1])) #converte la lista di tuple in una lista di liste
    
    # Remove boulders in the black list
    boulders_positions = [pos for pos in boulders_positions if pos not in set(black_list_boulder)]
    
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

def is_obstacle(position_element: int, coordinates : Tuple[int,int], target_position: Tuple[int,int], boulder_is_obstacle=False) -> bool:
    """
        checks if the element in the position is an obstacle
        :param position_element: the element to check
    """

    wall = "|- "
    boulder = "`"

    if boulder_is_obstacle and chr(position_element) == boulder:
        return True

    if coordinates == target_position:
        return True

    return chr(position_element) in wall # or chr(position_element) == river

def are_less_black_blocks(old_map, new_map):
    """
        old_map: the previous state of the game map
        new_map: the current state of the game map
        return: True iff the number of black blocks (unseen map) is fewer than the previous state
    """

    black_block = ord(" ") #integer rapresentation of an unseen block of map
    tuples = np.where(old_map == black_block)
    old_black_blocks = list(zip(tuples[0], tuples[1]))

    tuples = np.where(new_map == black_block)
    new_black_blocks = list(zip(tuples[0], tuples[1]))

    return old_black_blocks != new_black_blocks

def are_less_black_blocks_light(old_number_black_blocks, new_map):
    """
        old_number_black_blocks: the previous number of black blocks (unseen blocks)
        new_map: the current state of the game map
        return: True iff the number of black blocks (unseen map) is fewer than the previous state
    """

    black_block = ord(" ") #integer rapresentation of an unseen block of map
    tuples = np.where(new_map == black_block)
    new_black_blocks = list(zip(tuples[0], tuples[1]))

    return old_number_black_blocks > len(new_black_blocks)

def get_number_black_blocks(game_map):
    """
        game_map: the current state of the game map
        return: the number of black blocks (unseen map)
    """

    black_block = ord(" ") #integer rapresentation of an unseen block of map
    tuples = np.where(game_map == black_block)
    black_blocks = list(zip(tuples[0], tuples[1]))

    return len(black_blocks)

def get_valid_moves(game_map: np.ndarray, current_position: Tuple[int, int], target_position: Tuple[int,int],hasBoulder:bool, boulder_is_obstacle=False) -> List[Tuple[int, int]]:
    """
        gets all the valid moves the player can make from the current position
        :param game_map: the game map as a matrix
        :param current_position: the current position of the agent
        :param target_position: the target position of the agent
        :param hasBoulder: if the agent has a boulder
        :return: a list of valid moves  
    """

    x_limit, y_limit = game_map.shape
    valid = []
    x, y = current_position   
    river = "}" 
   
    
    # North valid move
    if y - 1 > 0 and not is_obstacle(game_map[x, y-1],current_position, target_position, boulder_is_obstacle):
        valid.append((x, y-1)) 
    
    # East valid move
    if x + 1 < x_limit and not is_obstacle(game_map[x+1, y], current_position, target_position, boulder_is_obstacle):
        if not hasBoulder and not chr(game_map[x+1, y]) == river:
            valid.append((x+1, y))
        elif hasBoulder:
            valid.append((x+1, y)) 
        
    # South valid move
    if y + 1 < y_limit and not is_obstacle(game_map[x, y+1], current_position,target_position, boulder_is_obstacle):
        valid.append((x, y+1)) 
    # West valid move
    if x - 1 > 0 and not is_obstacle(game_map[x-1, y], current_position,target_position, boulder_is_obstacle):
        valid.append((x-1, y))
    # North-East valid move
    if x + 1 < x_limit and y - 1 > 0 and not is_obstacle(game_map[x+1, y-1], current_position,target_position, boulder_is_obstacle):
        if not hasBoulder and not chr(game_map[x+1, y-1]) == river:
            valid.append((x+1, y-1))
        elif hasBoulder:
            valid.append((x+1, y-1))
    # South-East valid move
    if x + 1 < x_limit and y + 1 < y_limit and not is_obstacle(game_map[x+1, y+1], current_position,target_position, boulder_is_obstacle):
        if not hasBoulder and not chr(game_map[x+1, y+1]) == river:
            valid.append((x+1, y+1))
        elif hasBoulder:
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

def actions_from_path(start: Tuple[int, int], path: List[Tuple[int, int]]) -> Tuple[List[int], List[str]]:
    """
        gets all the actions the player has to make to follow a path
        :param start: the starting position of the player
        :param path: the path to follow
        :return: a list of actions to perform
    """
    
    actions = []
    action_name = []
    i_s, j_s = start
    for (i, j) in path:
        action, name = action_map((i_s, j_s), (i,j))

        actions.append(action)
        action_name.append(name)

        i_s = i
        j_s = j
    
    
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

def execute_actions(env: gym.Env ,game: np.ndarray , game_map : np.ndarray, actions : List[int]):
    """
    Executes a list of actions in the environment and returns the rewards obtained for each action.
    Without plotting.

    Parameters:
    - env (gym.Env): The environment in which the actions will be executed.
    - game (np.ndarray): The game state.
    - game_map (np.ndarray): The game map.
    - actions (List[int]): The list of actions to be executed.

    Returns:
    - rewards (List[float]): The rewards obtained for each action.
    """
    rewards = []
    for action in actions:
        s, r, _, _ = env.step(action)
        rewards.append(r)
    return rewards

def plot_animated_sequence(env: gym.Env ,game: np.ndarray , game_map : np.ndarray, actions : List[int]):
    """
    Plots an animated sequence of the game environment based on a sequence of actions.

    Args:
        env (gym.Env): The game environment.
        game (np.ndarray): The game state.
        game_map (np.ndarray): The game map.
        actions (List[int]): The sequence of actions to be performed.

    Returns:
        List[Tuple[int, int]]: The player positions at each step of the sequence.
    """
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
    #print("Rewards: ")
    for r in rewards:
        print(r)
    #print("Total reward: ", sum(rewards))
        
    return rewards

def plot_anim_seq_online_a_star(game : np.ndarray, image):
    #Plotting the image
    display.display(plt.gcf())
    display.clear_output(wait=True)
    image.set_data(game['pixel'][:, :])
    time.sleep(0.5)

def plot_and_save_sequence(gamestate : dict):
    # Create directory if it doesn't exist
    if not os.path.exists('results'):
        os.makedirs('results')

    # Create a subdirectory for this test
    test_id = datetime.now().strftime('%Y%m%d%H%M%S')
    test_dir = os.path.join('results', test_id)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    rewards = []
    image = plt.imshow(gamestate['game'][25:300, :475])
    player_positions = []

    for action in gamestate['actions']:
        s, r, _, _ = gamestate['env'].step(action)
        rewards.append(r)
        image.set_data(s['pixel'][:, :])
        player_positions.append(get_player_location(gamestate['game_map']))
        time.sleep(0.5)

    # Save the initial and final images
    plt.imsave(os.path.join(test_dir, 'start_img.png'), gamestate['game'][:, :])
    plt.imsave(os.path.join(test_dir, 'end_img.png'), s['pixel'][:, :])

    # Write player positions, actions, and rewards to logs.txt
    with open(os.path.join(test_dir, 'logs.txt'), 'w') as f:

        f.write("Agent starting position:\n")
        f.write(str(gamestate['start']) + "\n\n")

        f.write("Boulders list:\n")
        f.write(str(gamestate['boulders_list']) + "\n\n")

        f.write("Coordinates min boulder:\n")
        f.write(str(gamestate['coordinates_min_boulder']) + "\n\n")

        f.write("Final positions:\n")
        f.write(str(gamestate['final_position']) + "\n\n")

        f.write("River positions:\n")
        f.write(str(gamestate['river_positions']) + "\n\n")

        f.write("Path player to pushing position:\n")
        f.write(str(gamestate['path_player_to_pushing_position']) + "\n\n")

        f.write("Path boulder to river:\n")
        f.write(str(gamestate['path_boulder_river']) + "\n\n")

        f.write("Full path of the agent:\n")
        f.write(str(gamestate['agent_full_path']) + "\n\n")

        f.write("Actions:\n")
        f.write(str(gamestate['actions']) + "\n")
        f.write(str(gamestate['names']) + "\n\n")

        f.write("Player Positions:\n")
        f.write(str(player_positions) + "\n\n")

        f.write("Rewards:\n")
        f.write(str(rewards) + "\n")
        f.write("\nTotal Reward: " + str(sum(rewards)))

    return player_positions

def plot_avg_steps_difference(array1, array2):
    """
    Plots the average steps difference between two arrays.
    
    Parameters:
    array1 (array-like): First array containing step values.
    array2 (array-like): Second array containing step values.
    
    Returns:
    None
    """
    labels = ['Avg optimal step', 'Avg online a star steps']
    sums = [np.sum(array1) / len(array1), np.sum(array2) / len(array1)]
    
    colors = ['blue', 'red']  # Specify the colors for the bars
    
    plt.bar(labels, sums, color=colors)
    plt.ylabel('Total number of steps')
    plt.title('A star vs Online A star: steps')
    plt.show()

def plot_success_rate(a_star_success : int, online_a_star_success : int, tot : int):
    """
    Plot the success rate of A star and Online A star algorithms.
    
    Parameters:
    a_star_success (int): The number of successful runs for A star algorithm.
    online_a_star_success (int): The number of successful runs for Online A star algorithm.
    tot (int): The total number of runs.
    """
 
    labels = ['A star', 'Online A star']
    values = [a_star_success,online_a_star_success]
    
    colors = ['blue', 'red']  # Specify the colors for the bars
    
    plt.bar(labels, values, color=colors)
    plt.ylabel('% of success ')
    plt.title('A star vs online A star: Success rate')
    plt.show()

def compute_percentage_difference(array1, array2):
    """
    Computes the percentage difference between the sums of two arrays.
    Usefull to get how many steps more we do with online a star compared to a star.

    Parameters:
    array1 (numpy.ndarray): The first array.
    array2 (numpy.ndarray): The second array.

    Returns:
    float: The percentage difference between the sums of the two arrays.
    """
    sum1 = np.sum(array1)
    sum2 = np.sum(array2)
    
    difference = abs(sum1 - sum2)
    percentage_difference = (difference / ((sum1 + sum2) / 2)) * 100
    
    return round(percentage_difference, 2)

def new_init_(self, *args, **kwargs):
    """
    Initialize the environment

    Parameters:
    args: arguments
    kwargs: keyword arguments
    """

    kwargs["max_episode_steps"] = kwargs.pop("max_episode_steps", 350)
    n_monster = kwargs.pop("n_monster", 0)
    n_boulder = kwargs.pop("n_boulder", 5)
    narrow = kwargs.pop("narrow", False)
    lava = kwargs.pop("lava", False)

    if narrow:
        map = """
....................W....
....................W....
....................W....
....................W....
....................W....
....................W....
....................W....
"""
    elif not lava:
        map = """
..................WWW....
..................WWW....
..................WWW....
..................WWW....
..................WWW....
..................WWW....
..................WWW....
"""
    else:
        map = """
..................LLL....
..................LLL....
..................WWW....
..................LLL....
..................WWW....
..................LLL....
..................LLL....
"""

    lvl_gen = LevelGenerator(map=map)
    lvl_gen.set_start_rect((0, 0), (18, 6))

    for _ in range(n_monster):
        lvl_gen.add_monster()

    lvl_gen.set_area_variable(
        "$boulder_area", type="fillrect", x1=1, y1=1, x2=18, y2=5
    )
    for _ in range(n_boulder):
        lvl_gen.add_object_area(
            "$boulder_area", name="boulder", symbol="`"
        )

    lvl_gen.add_goal_pos((24, 2))

    MiniHackNavigation.__init__(self, *args, des_file=lvl_gen.get_des(), **kwargs)

def create_env(seeds=[]) -> Tuple[dict, gym.Env]:
    minihack.envs.river.MiniHackRiver.__init__ = new_init_  # Aggiornamento mappa river
    if len(seeds) == 0:
        env = gym.make("MiniHack-River-Narrow-v0", observation_keys=("chars", "pixel", "colors"))
        state = env.reset()
    else:
        env = gym.make("MiniHack-River-Narrow-v0", observation_keys=("chars", "pixel", "colors"), seeds=seeds)
        state = env.reset()

    return state, env
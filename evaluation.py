from algorithms import *
from logic import *
from utils import *

def test_online_a_star(iterations, tests):
    """
    Test the A* algorithm online.

    Args:
        iterations (int): The number of iterations to run the test.
        tests (int): The number of tests to perform.

    Returns:
        None
    """
    print("\nStarting test: A* online\n")
    list_completed = []
    list_avg_time_per_success = []
    list_avg_steps = []
    
    for j in range(tests):
        successfully_completed = 0
        number_of_steps = []
        avg_time = []
        
        for i in range(iterations):
            run_steps = 0
            black_list_boulders = []
            env = gym.make("MiniHack-River-Narrow-v0", observation_keys=("chars", "pixel"))
            state = env.reset()
            
            start_time = time.time()
            
            obs, river_target, steps = push_one_boulder_into_river(state, env, black_list_boulders, plot=False)
            
            if obs is None and river_target is None and steps is None:
                continue

            while river_target is None or obs['chars'][river_target] == ord("}"):
                if obs is None:
                    break
                obs, river_target, steps = push_one_boulder_into_river(obs, env, black_list_boulders, plot=False)
                if(steps is not None):
                    run_steps = run_steps + steps
                
            if obs is None:
                continue
            
            run_steps = run_steps + steps
            game_map = state['chars']
            #Going between the river
            action,_  = action_map(get_player_location(game_map), river_target)
            state,_,_,_ = env.step(action)
            run_steps = run_steps + 1

            result,steps = find_exit(env, game_map)

            if result == 1.0:
                elapsed_time = time.time() - start_time
                successfully_completed = successfully_completed + 1
                avg_time.append(elapsed_time)
                number_of_steps.append(run_steps+steps)

        print("Successfully completed: ", successfully_completed,"/",iterations, " = ", successfully_completed/iterations*100, "%")
        print("Average number of steps: ", sum(number_of_steps)/successfully_completed)
        print("Average time per success: ", sum(avg_time) / successfully_completed, "\n")
        list_completed.append(successfully_completed)
        list_avg_time_per_success.append(sum(avg_time) / successfully_completed)
        list_avg_steps.append(sum(number_of_steps)/successfully_completed)

    print("Average total succesfully completed: ", (sum(list_completed)/tests)/iterations*100, "%")
    print("Average total steps: ", sum(list_avg_steps)/tests)
    print("Average total time: ", sum(list_avg_time_per_success)/tests, "\n")


def test_classic_a_star(iterations, tests):
    """
    Test the A* classic algorithm for crossing the river in a MiniHack environment.

    Args:
        iterations (int): The number of iterations to run the test.
        tests (int): The number of tests to perform.

    Returns:
        None
    """
    list_completed = []
    list_avg_time_per_success = []
    list_avg_steps = []

    print("\nStarting test: A* classic\n")
    # Rest of the code...
def test_classic_a_star(iterations,tests):
    list_completed = []
    list_avg_time_per_success = []
    list_avg_steps = []


    print("\nStarting test: A* classic\n")
    for j in range(tests):
        successfully_completed = 0
        number_of_steps = []
        avg_time = []

        
        for i in range(iterations):
            run_steps = 0
            black_list_boulders = []
            env = gym.make("MiniHack-River-Narrow-v0", observation_keys=("chars", "pixel"))
            state = env.reset()
            start_time = time.time()
            game_map = state['chars']
            game = state['pixel']

            
            #get locations of the player, boulders and river
            start = get_player_location(game_map)
            boulders = get_boulder_locations(game_map, black_list_boulders)
            river_positions = find_river(env, game_map, black_list_boulders)
            target = None

            #Get full path from the player to the river
            
            coordinates_min_boulder = get_best_global_distance(start, boulders, river_positions)
            temp = get_min_distance_point_to_points(coordinates_min_boulder[0],coordinates_min_boulder[1], river_positions)
            final_position = tuple(temp[0])
            

            hasBoulder = True #The river is not considered as an obstacle

            #Calculating the path from the boulder to the river shortest distance
            path_boulder_river = a_star(game_map, coordinates_min_boulder,final_position, hasBoulder, False, get_optimal_distance_point_to_point)
            

            #Calculating the position in which the agent have to be in order to push correctly the boulder into the river
            pushing_position = position_for_boulder_push(coordinates_min_boulder, path_boulder_river[1])[1]
            
            
            nearest_pushing_position = pushing_position
            
            if game_map[pushing_position] == ord(" "): # the target is an unseen block
                continue

            else:
                hasBoulder = False #The river is considered as an obstacle
                #Calculating the path from the player to the pushing position
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
            
            if agent_full_path is None:
                continue    

            actions, names = actions_from_path(start, agent_full_path) 
            
            actions = actions[1:] #Remove first action because it is the start position
            
            # payer_pos -> river 
            for action in actions:
                s, r, _, _ = env.step(action)
                run_steps= run_steps + 1
                #time.sleep(0.5)
            
            # step to the river
            current_river_target = path_boulder_river[-1]
            player_pos = get_player_location(game_map)
            action, name = action_map(player_pos, current_river_target)
            obs, _,_,_ = env.step(action)
            run_steps = run_steps + 1

            game_map = obs['chars']
            game_gui = obs['pixel']

            player_pos = get_player_location(game_map)
            exit_pos = get_exit_location(game_map)
            # river -> stairs (exit)
            game_map = obs['chars']
            
            result,steps = find_exit(env, game_map)
            run_steps = run_steps + steps

            if result == 1.0:
                elapsed_time = time.time() - start_time
                successfully_completed = successfully_completed + 1
                avg_time.append(elapsed_time)
                number_of_steps.append(run_steps)
            
        if successfully_completed == 0:
            print("No success = 0%")
            list_completed.append(successfully_completed)
            list_avg_time_per_success.append(0)
            list_avg_steps.append(0)
        else:
            print("Successfully completed: ", successfully_completed,"/",iterations, " = ", successfully_completed/iterations*100, "%")
            print("Average number of steps: ", sum(number_of_steps)/successfully_completed)
            print("Average time per success: ", sum(avg_time) / successfully_completed, "\n")
            list_completed.append(successfully_completed)
            list_avg_time_per_success.append(sum(avg_time) / successfully_completed)
            list_avg_steps.append(sum(number_of_steps)/successfully_completed)

    print("Average total succesfully completed: ", (sum(list_completed)/tests)/iterations*100, "%")
    print("Average total steps: ", sum(list_avg_steps)/tests)
    print("Average total time: ", sum(list_avg_time_per_success)/tests, "\n")


if __name__ == "__main__":
    #test_online_a_star(300, 10)
    test_classic_a_star(300,10)
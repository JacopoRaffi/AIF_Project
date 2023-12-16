from algorithms import *
from logic import *
from utils import *

def test_online_a_star(iterations):
    successfully_completed = 0
    number_of_steps = []

    for i in range(iterations):
        print("Iteration: ", i)
        run_steps = 0
        env = gym.make("MiniHack-River-Narrow-v0", observation_keys=("chars", "pixel", "colors"))
        state = env.reset()

        game_map = state['chars']
        obs, river_target, steps = push_one_boulder_into_river(state, env)

        print("River target: ", river_target)
        print(state['chars'][river_target[0]][river_target[1]])
        print(ord("}"))

        
        while obs['chars'][river_target] == ord("}"):
            obs, river_target, steps = push_one_boulder_into_river(state, env)
            run_steps = run_steps + steps
        
        if(river_target is None): #For now we will just skip this iteration and consider it a fail
            continue

        run_steps = run_steps + steps

        action, _ = action_map(get_player_location(game_map), river_target)
        state,_,_,_ = env.step(action)

        game_map = state['chars']

        result = find_exit(env, game_map)

        if result == 1.0:
            successfully_completed = successfully_completed + 1
            number_of_steps.append(run_steps)


    print("Successfully completed: ", successfully_completed,"/",iterations)
    print("Average number of steps: ", sum(number_of_steps)/iterations)

if __name__ == "__main__":
    test_online_a_star(1000)
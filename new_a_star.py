def a_star(game : np.ndarray, game_map: np.ndarray, start: Tuple[int, int], target: Tuple[int, int], h: callable) -> List[Tuple[int, int]]:
    count = 0 #When an item is inserted into the queue, if there is a tie choses the one inserted first
    open_set = PriorityQueue()
    open_set.put((0, count, start)) #f score, count, position
    came_from = {} #Tracks the path
    g_scores = {} #Dictionary with the g score of each node
    f_scores = {} #Dictionary with the f score of each node

    map_positions = get_map_positions(game_map) #Get all the positions x,y in the map

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
    #Setta il g_score di tutti i nodi della mappa a infinito nel dizionario
   
    #Seta l'f score di tutti i nodi della mappa a infinito nel dizionario
    open_set_hash = {start} #Track all the items in the priority queue

    while not open_set.empty():
        current = open_set.get()[2] #Get the position of the current node, get the item with the lowest f score
        open_set_hash.remove(current) #Remove the current node from the open set

        if current == target:
            return True
        
        for neighbour in get_valid_moves(game_map, current): #Neighbours of the current node
            
            temp_g_score = g_scores[current] + 1 #g score of the neighbour calulated as the g score of the current node + 1

            if temp_g_score < g_scores[neighbour]: #if we found a better way to reach this neighbour update the g score
                came_from[neighbour] = current #Update the node from which we reached the neighbour
                g_scores[neighbour] = temp_g_score #Update the g score of the neighbour
                f_scores[neighbour] = temp_g_score + h(neighbour, target) #Update the f score of the neighbour

                if neighbour not in open_set_hash:
                    count += 1
                    open_set.put((f_scores[neighbour], count, neighbour)) #Add the neighbour to the open set because it is the best path to reach the target
                    open_set_hash.add(neighbour)
    return False    

def reconstruct_path(came_from, current):
    path = []
    while current in came_from:
        current = came_from[current]
        path.append(current)
    return path.reverse()

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
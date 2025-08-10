import numpy as np
import matplotlib.pyplot as plt
import heapq
import io
import sys
import time 

N = 8  # Size of the grid (N x N)
DIRS = [(-1,0),(1,0),(0,-1),(0,1)]  # 4-way movement directions: 
                                    # up, down, left, right

def raw_map(prob_block=0.0, start=None, goal=None):
    """
    Generate a random N x N grid for pathfinding with obstacles.

    Each cell is independently set to either open (0) or blocked (1), based on 
    prob_block.
    Ensures start and goal positions are open, and each has at least one open 
    neighbor (to avoid dead-ends).

    Parameters:
        prob_block (float): Probability that each cell (except start/goal) is 
                            an obstacle (blocked).
        start (tuple): (row, col) index for start position (guaranteed open).
        goal (tuple): (row, col) index for goal position (guaranteed open).

    Returns:
        grid (np.ndarray): 2D array of shape (N, N), with 0=open and 1=obstacle.
    """
    if start is None or goal is None:
        raise ValueError("start and goal must be specified")
    
    # Randomly assign 0 (open) or 1 (blocked) for each cell
    grid = np.random.choice([0, 1], size=(N, N), p=[1-prob_block, prob_block])

    # Guarantee that start and goal cells are open, never blocked
    # Decrease the chance to have no completed path
    grid[start[0], start[1]] = 0
    grid[goal[0], goal[1]] = 0

    # Guarantee at least one open neighbor for start
    start_neighbors = []
    for dx, dy in DIRS:
        nx, ny = start[0]+dx, start[1]+dy
        if 0 <= nx < N and 0 <= ny < N:
            start_neighbors.append((nx, ny))
    np.random.shuffle(start_neighbors)  # Randomize neighbor order
    neighbor_opened = any(grid[nx, ny] == 0 for nx, ny in start_neighbors)
     # If no open neighbor, forcibly open the first neighbor
    if not neighbor_opened and start_neighbors:
        nx, ny = start_neighbors[0]
        grid[nx, ny] = 0

    # Guarantee at least one open neighbor for goal
    goal_neighbors = []
    for dx, dy in DIRS:
        nx, ny = goal[0]+dx, goal[1]+dy
        if 0 <= nx < N and 0 <= ny < N:
            goal_neighbors.append((nx, ny))
    np.random.shuffle(goal_neighbors)
    neighbor_opened = any(grid[nx, ny] == 0 for nx, ny in goal_neighbors)
    if not neighbor_opened and goal_neighbors:
        nx, ny = goal_neighbors[0]
        grid[nx, ny] = 0

    return grid

def heuristic(a, b):
    """
    Compute the Manhattan distance between two cells.

    Used as the A* heuristic: h(n) = |x1-x2| + |y1-y2|

    Parameters:
        a, b (tuple): Cells as (row, col)

    Returns:
        int: Manhattan distance between a and b
    """
    return abs(a[0]-b[0]) + abs(a[1]-b[1])


def build_path(came_from, start, goal):
    """
    Reconstructs the path from goal to start using the came_from map.

    Parameters:
        came_from (dict): Mapping of node -> parent node.
        start (tuple): Start node (row, col).
        goal (tuple): Goal node (row, col).

    Returns:
        path (list): List of nodes from start to goal, or None if unreachable.
    """
    path = [goal]
    current = goal
    while current != start:
        if current not in came_from:
            return None  # No path found
        current = came_from[current]
        path.append(current)
    return path[::-1]


def a_star(grid, start, goal, return_costs=False, return_expanded=False, 
           return_visited=False):
    """
    Performs the A* pathfinding algorithm on a 2D grid with obstacles, 
    finding an optimal path from start to goal.

    The implementation uses a parent-pointer dictionary (came_from) to rebulid 
    the shortest path after search, minimizing memory usage.

    Tie-breaking in the priority queue uses the heuristic value (h) to prefer 
    nodes closer to the goal when f values are equal.

    Parameters:
        grid: np.ndarray
              The 2D grid representing the environment 
              (0=open cell, 1=blocked cell).
        start: tuple
               (row, col) coordinates for the starting node.
        goal: tuple
              (row, col) coordinates for the goal node.
        return_costs: bool, optional
                      If True, returns a dictionary mapping expanded nodes to 
                      their (g, h, f) costs.
        return_expanded: bool, optional
                         If True, returns the total number of unique nodes 
                         expanded during the search.
        return_visited: bool, optional
                        If True, returns a list representing the exact order 
                        in which nodes were expanded.

    Returns:
        tuple (variable-length)
            The first returned value is always:
            - path: list of (row, col) tuples representing the optimal path 
                     from start to goal, or None if no path is found.

            Optional returned values based on flags:
            - costs: dict mapping (row, col) -> (g, h, f) for each expanded node
                     (if return_costs=True).
            - expanded: int, total number of unique nodes expanded 
                        (if return_expanded=True).
            - visit_order: list of (row, col) tuples showing the order of node 
                           expansions (if return_visited=True).

    Optimization Notes:
        - Use a min-heap (priority queue) and a heuristic-based tie-breaking 
          method: prioritizing nodes with lower heuristic values (closer to 
          the goal) when nodes have identical f(n) values.
        - This tie-breaking reduces unnecessary expansions in grids with uniform 
          movement costs and identical values.
        - The parent-pointer method allows efficient O(N) memory use,
          rebuilding the full path only once after the search.

    Complexity:
        - Best-case: O(N logN)
        - Average-case: <= O(N^2 logN)
        - Worst-case: O(N^2 logN)
    """
    open_set = []       # priority queue of nodes to explore (frontier nodes)
    closed_set = set()  # Set of already-expanded nodes to avoid re-exploration

    came_from = {}      # To reconstruct the optimal path at the end
    
    # Dictionary tracking the lowest cost (g) for each node
    g_score = {start: 0}
    
    # Dictionary tracking costs (g, h, f) for each node for return purposes
    costs = {}

    visit_order = []            # List tracking the order of nodes expanded
    expanded_nodes_count = 0    # Counter tracking total number of expansions

    # Compute heuristic (Manhattan distance) from start to goal
    h_start = heuristic(start, goal)
    f_start = h_start  # g(start) = 0. So, f(start) = h(start)
    if return_costs:
        costs[start] = (0, h_start, f_start)

    # Push the start node onto the heap; tuple is (f, h, node)
    heapq.heappush(open_set, (f_start, h_start, start))

    while open_set:
        # Extract node with lowest f (break ties with h)
        current_f, current_h, current_node = heapq.heappop(open_set)

        # # Skip node if already expanded (could be duplicate in heap)
        if current_node in closed_set:
            continue

        # Mark as expanded
        closed_set.add(current_node)

        # Record expansion for optional statistics
        if return_visited:
            visit_order.append(current_node)
        if return_expanded:
            expanded_nodes_count += 1

        # Check Goal is found
        if current_node == goal:
            # Rebuild the optimal path from start to goal
            path = build_path(came_from, start, goal)
            result = [path]
            if return_costs:
                result.append(costs)
            if return_expanded:
                result.append(expanded_nodes_count)
            if return_visited:
                result.append(visit_order)
            return tuple(result)

        # Explore neighbors in 4 directions
        for direction in DIRS:
            neighbor_row = current_node[0] + direction[0]
            neighbor_col = current_node[1] + direction[1]
            neighbor_node = (neighbor_row, neighbor_col)

            # Skip neighbor if out of bounds or blocked
            if not (0 <= neighbor_row < N and 
                    0 <= neighbor_col < N and 
                    grid[neighbor_row, neighbor_col] == 0):
                continue

            current_g = g_score[current_node] # Cost (g) to reach current node
            tentative_g = current_g + 1       # Cost (g) to reach neighbor

            # Skip neighbor if no improvement in cost is found
            if neighbor_node in g_score and tentative_g >= g_score[neighbor_node]:
                continue

            # Record optimal parent and cost for neighbor
            came_from[neighbor_node] = current_node
            g_score[neighbor_node] = tentative_g

            # Compute heuristic and total cost for neighbor
            neighbor_h = heuristic(neighbor_node, goal)
            neighbor_f = tentative_g + neighbor_h

            if return_costs:
                costs[neighbor_node] = (tentative_g, neighbor_h, neighbor_f)

            # Insert neighbor into priority queue with heuristic tie-breaking
            # (sort by f, then h)
            heapq.heappush(open_set, (neighbor_f, neighbor_h, neighbor_node)) 
                                      
    # No path found: return None and optionally other requested data
    result = [None]
    if return_costs:
        result.append(costs)
    if return_expanded:
        result.append(expanded_nodes_count)
    if return_visited:
        result.append(visit_order)
    return tuple(result)


def plot_map_with_costs(grid, costs, path=None, title="", start=None, goal=None,
                       filename=None, visited=None):
    """
    Visualizes the 2D grid:
    - Obstacles in black, open cells white
    - Visited/expanded nodes as yellow squares (no order numbers)
    - Solution path in red
    - Start (green star), Goal (blue star)
    - g/h/f values overlaid on expanded nodes

    Parameters:
        grid (np.ndarray): 2D grid (0=open, 1=blocked)
        costs (dict): (row, col) -> (g, h, f)
        path (list): Path from start to goal (optional)
        title (str): Plot title
        start (tuple): Start node (row, col)
        goal (tuple): Goal node (row, col)
        filename (str): If given, save figure to this file
        visited (list/set): Expanded nodes to highlight (optional)

    Returns:
        None. Shows and optionally saves the plot.
    """
    if start is None or goal is None:
        raise ValueError("start and goal must be specified")
    plt.figure(figsize=(N,N))
    plt.title(f"{title}\n(g: steps from start, h: heuristic to goal, f: g+h)")
    plt.imshow(grid, cmap='gray_r')
    plt.xticks(np.arange(grid.shape[1]))
    plt.yticks(np.arange(grid.shape[0]))
    plt.grid(True, color='lightgray')

    # Plot expanded/visited nodes as yellow squares
    if visited:
        vx, vy = zip(*visited)
        plt.scatter(vy, vx, c='orange', s=90, marker='s', alpha=0.5, 
                    label='Visited')

    # Overlay g/h/f with labels for expanded nodes
    for (x, y), (g, h, f) in costs.items():
        txt = f"g={g}\nh={h}\nf={f}"
        plt.text(
            y, x, txt, ha='center', va='center', fontsize=8, color='black',
            bbox=dict(facecolor='white', alpha=0.65, edgecolor='none', 
                      boxstyle='round,pad=0.18'))

    # Draw the solution path
    if path:
        px, py = zip(*path)
        plt.plot(py, px, color='red', linewidth=3, marker='o', markersize=8, 
                 label='Path')

    # Draw start and goal
    plt.scatter([start[1]], [start[0]], color='green', s=200, marker='*', 
                edgecolors='black', linewidths=1, zorder=2, label='Start')
    plt.scatter([goal[1]], [goal[0]], color='blue', s=200, marker='*', 
                edgecolors='black', linewidths=1, zorder=2, label='Goal')

    # Legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(
        by_label.values(), by_label.keys(),
        loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0)
    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename, dpi=180, bbox_inches='tight')
    plt.show()
    plt.close()


def print_path_costs_and_heuristics(path, goal, algo='astar'):
    """
    Prints the actual cost (g), heuristic (h), and f value for each cell 
    along the found path.
    For Dijkstra, h is always 0, f = g.
    For A*, h is Manhattan distance, f = g + h.

    Parameters:
        path: list of tuple
              List of (row, col) tuples along the found path.
        goal: tuple
              (row, col) tuple, the goal cell (for heuristic computation).

    Returns:
        None
    """
    print("Actual cost (g) and heuristic (h) values along the path:")
    for idx, cell in enumerate(path):
        g = idx  # Each step costs exactly 1
        h = heuristic(cell, goal)
        f = g + h
        print(f"  {cell}: g={g}, h={h}, f={f}")
    print()


def print_grid_stats(grid):
    """
    Prints statistics about the generated grid:
      - Total number of cells
      - Number and percentage of obstacles
      - Number and percentage of free cells

    Parameters:
        grid: np.ndarray
              2D grid of the map (0=open, 1=obstacle).

    Returns:
        None    
    """
    total = grid.size
    obstacles = np.sum(grid == 1)
    free = np.sum(grid == 0)
    percent_obstacles = 100.0 * obstacles / total
    percent_free = 100.0 * free / total
    print(f"Total cells: {total}")
    print(f"Obstacles: {obstacles} ({percent_obstacles:.1f}%)")
    print(f"Free cells: {free} ({percent_free:.1f}%)\n")


def create_map(prob_block=0.0, max_attempts=1000, start=None, goal=None):
    """
    Generate a random map with a valid path from start to goal.

    - Tries up to max_attempts times to produce a map where a valid path exists.
    - Uses A* to check for solvability.
    - If no path is found after all attempts, raises an error.

    Parameters:
        prob_block: float
                    Probability that a given cell is an obstacle.
        start: tuple
               (row, col) position for the start node.
        goal: tuple
              (row, col) position for the goal node.
        max_attempts: int, optional
                      Maximum number of attempts to generate a solvable map.

    Returns:
        grid: np.ndarray
              2D grid with 0=open, 1=obstacle.
        path: list of tuple
              The shortest path from start to goal, if found.
        costs: dict
               Expanded cells and their (g, h, f) values.

    Raises:
        RuntimeError
            If no solvable map is found after max_attempts.
    """
    if start is None or goal is None:
        raise ValueError("start and goal must be specified")
    for attempt in range(max_attempts):
        grid = raw_map(prob_block=prob_block, start=start, goal=goal)
        path, costs = a_star(grid, start, goal, return_costs=True, 
                             return_visited=False)
        if path:
            return grid, path, costs
    raise RuntimeError(
        f"Could not create a solvable map in {max_attempts} attempts. "
        "Try lowering obstacle rate or increasing attempts.")


def main():
    """
    For each test case (obstacle probability), generates a random map,
    runs A* algorithm, and prints/saves:
      - The found path
      - Number of expanded nodes and runtime
      - The full list of expanded (visited) nodes in order
      - Step-by-step g/h/f values along the path
    Visualizations show the path and expansion order.
    All output is saved to a .txt file for each case.
    """
    # np.random.seed(42) # For reproducibility: Remove the comment sign (#)

    start = (0, 0)
    goal = (7, 7)
    probs = [0.0, 0.2, 0.5]
    titles = ["No Obstacles (0%)", "20% Obstacles", "50% Obstacles"]

    for i, (prob, title) in enumerate(zip(probs, titles)):
        output_lines = []
        print(f"\n=== Case: {title} ===")
        output_lines.append(f"\n=== Case: {title} ===")
        print(f"Start: {start}   Goal: {goal}")
        output_lines.append(f"Start: {start}   Goal: {goal}")

        try:
            # Generate a random solvable map for this case
            grid, _, _ = create_map(prob_block=prob, start=start, goal=goal)

            # Run A* time
            t0 = time.time()
            path_astar, costs_astar, astar_expanded, astar_visit_order = a_star(
                grid, start, goal, return_costs=True, return_expanded=True, 
                return_visited=True)
            t1 = time.time()
            astar_time = t1 - t0

            # Print map statistics
            tempbuf = io.StringIO()
            old_stdout = sys.stdout
            try:
                sys.stdout = tempbuf
                print_grid_stats(grid)
            finally:
                sys.stdout = old_stdout
            gridstats = tempbuf.getvalue().rstrip()
            print(gridstats)
            output_lines.append(gridstats)
            print()
            output_lines.append("")

            # A* path
            print("A* Path found:")
            output_lines.append("A* Path found:")
            print(path_astar)
            output_lines.append(str(path_astar))
            print()
            output_lines.append("")

            print(f"A* expanded {astar_expanded} nodes, "
                  f"time: {astar_time:.6f} sec")
            output_lines.append(
                f"A* expanded {astar_expanded} nodes, "
                f"time: {astar_time:.6f} sec")
            
            if path_astar:
                print(f"Total nodes in found path: {len(path_astar)}")
                output_lines.append(f"Total nodes in found path: {len(path_astar)}")

            print()
            output_lines.append("")

            # Print the full expansion order for A*
            print("A* visited node order (all expanded nodes):")
            output_lines.append("A* visited node order (all expanded nodes):")
            for idx, cell in enumerate(astar_visit_order):
                print(f"  {idx+1:3}: {cell}")
                output_lines.append(f"  {idx+1:3}: {cell}")
            print()
            output_lines.append("")

            if path_astar:
                tempbuf = io.StringIO()
                old_stdout = sys.stdout
                try:
                    sys.stdout = tempbuf
                    print_path_costs_and_heuristics(path_astar, goal, 
                                                    algo='astar')
                finally:
                    sys.stdout = old_stdout
                pathcosts = tempbuf.getvalue().rstrip()
                print(pathcosts)
                output_lines.append(pathcosts)
                print()
                output_lines.append("")

            img_filename_astar = f"astar_case_{i+1}.png"
            plot_map_with_costs(grid, costs_astar, path_astar, 
                                title=title + " (A*)", start=start, goal=goal, 
                                filename=img_filename_astar, 
                                visited=astar_visit_order)

        except RuntimeError as e:
            errmsg = str(e)
            print(errmsg)
            output_lines.append(errmsg)
            print()
            output_lines.append("")

        # Save all outputs for this test case to a text file
        text_filename = f"astar_case_{i+1}.txt"
        with open(text_filename, "w", encoding="utf-8") as txtfile:
            for line in output_lines:
                txtfile.write(line + "\n")


if __name__ == "__main__":
    main()


'''
==============================================================================================
                                    Pseudocode 1 (Concise)
==============================================================================================  
Inputs: Grid of size N * N (0=open cell, 1=blocked cell)
        Start node (start)
        Goal node (goal)

Outputs: Optimal path from start to goal if one exists, else null.


Algorithm AStar(grid, start, goal):
    open_set := priority queue ordered by (f, h)      // frontier nodes to explore
    closed_set := empty set                           // tracks expanded nodes

    came_from := empty map                            // node -> parent (for path)
    g_score[start] := 0                               // cost from start to node

    h_start := HEURISTIC(start, goal)
    push open_set with (f = h_start, h = h_start, node = start)

    while open_set is not empty:
        (current_f, current_h, current_node) := pop from open_set

        if current_node in closed_set:
            continue      // skip already expanded node

        if current_node == goal:
            return BUILD_PATH(came_from, start, goal)

        add current_node to closed_set

        for each direction in [up, down, left, right]:
            neighbor := current_node moved in direction

            if neighbor is out of grid bounds or is blocked in grid:
                continue

            tentative_g := g_score[current_node] + 1

            if neighbor in g_score and tentative_g >= g_score[neighbor]:
                continue

            came_from[neighbor] := current_node
            g_score[neighbor] := tentative_g

            neighbor_h := HEURISTIC(neighbor, goal)
            neighbor_f := tentative_g + neighbor_h

            push open_set with (neighbor_f, neighbor_h, neighbor)

    // No path found
    return None
        
// Heuristic (Manhattan distance): estimates distance from current node to goal
Function HEURISTIC(node, goal):
    return |node.x - goal.x| + |node.y - goal.y|

// Reconstructs the path from goal to start using the came_from map 
Function BUILD_PATH(came_from, start, goal):
    path := list containing goal
    current := goal
    while current != start:
        if current not in came_from:
            return None                 // path not found
        current := came_from[current]
        append current to path
    reverse path
    return path

Note: 
- open_set: Min-heap/priority queue; always picks node with lowest estimated 
            total cost (f).
- came_from: Stores parent links for path reconstruction (memory-efficient).
- g_score: Dictionary of best-known costs to each node.
- Heuristic: Manhattan distance for grid.
- BUILD_PATH: Traces from goal to start using parents, then reverses to get 
              start -> goal order.
              
===============================================================================================
                                    Pseudocode 2 (Detailed)
===============================================================================================  
Inputs: Grid of size N * N (0=open cell, 1=blocked cell)
        Start node (start)
        Goal node (goal)

Outputs: Optimal path from start to goal if one exists, else null.


Algorithm AStar(grid, start, goal):
    open_set := empty min-heap priority queue
        // Stores tuples (f, h, node), where:
        //   f: total estimated cost (g + h)
        //   h: heuristic estimate from node to goal
        //   node: (row, col) cell coordinates

    closed_set := empty set
        // Tracks nodes already expanded

    came_from := empty map
        // For each node, stores the parent node (to reconstruct path)

    g_score := map with default value infinity
        // For each node, stores the lowest cost to reach from start
    g_score[start] := 0

    h_start := HEURISTIC(start, goal)
    f_start := h_start  // since g(start) = 0

    insert (f_start, h_start, start) into open_set

    while open_set is not empty do:
        (current_f, current_h, current) := open_set.pop_min()
            // Select node with minimal f; break ties with lower h

        if current = goal then:
            return BUILD_PATH(came_from, start, goal)

        if current in closed_set then:
            continue   // Skip already expanded nodes

        add current to closed_set

        for each direction in [up, down, left, right] do:
            neighbor := current moved in direction

            if not IN_BOUNDS(grid, neighbor) or grid[neighbor] = 1 then:
                continue   // Skip out-of-bounds or blocked cells

            tentative_g := g_score[current] + 1
                // All moves cost 1

            if neighbor ∈ g_score and tentative_g ≥ g_score[neighbor] then:
                continue   // Not a better path

            came_from[neighbor] := current
            g_score[neighbor] := tentative_g

            neighbor_h := HEURISTIC(neighbor, goal)
            neighbor_f := tentative_g + neighbor_h

            insert (neighbor_f, neighbor_h, neighbor) into open_set

    // If goal was never reached:
    return None
    
Function HEURISTIC(node, goal):
// Manhattan distance for grid-based movement
return |node.row - goal.row| + |node.col - goal.col|

Function BUILD_PATH(came_from, start, goal):
    path := list containing goal
    current := goal
    while current != start do:
        if current not in came_from:
            return None  // No path exists
        current := came_from[current]
        insert current at front of path
    return path

Function IN_BOUNDS(grid, node):
    // Returns True if node is inside the grid boundaries
    rows := number of rows in grid
    cols := number of columns in grid
    (r, c) := node
    return 0 ≤ r < rows and 0 ≤ c < cols

****************************************** [ Note ] *******************************************

1. Priority Queue Ordering and Tie-breaking:
Nodes are prioritized based on their estimated total cost f(n) = g(n) + h(n). 
When nodes have identical f(n) values, tie-breaking is performed using 
the heuristic value h(n), preferring nodes closer to the goal. 
This reduces unnecessary node expansions significantly.

2. Data Structures:
- openSet: Implemented as a min-heap priority queue, 
           extracting the minimal priority element in O(log N) time.
- closedSet: Tracks already expanded nodes, ensuring each node is expanded 
             at most once, for computational efficiency.

3. Parent pointer:
- The came_from map is used to reconstruct the path from the goal back to 
  the start after the search completes.

Time Complexity (N = Side length of the grid (number of rows/columns)):
Best-case: O(N logN)
Average-case: <= O(N^2 logN)
Worst-case: O(N^2 logN)

Spcae Complexity: O(N^2) <- Improved from previous version (Path in queue)
                            Best: O(N^2)/ Worst: O(N^4)
                            - open_set stores a full path(list) for each queue entry 

'''

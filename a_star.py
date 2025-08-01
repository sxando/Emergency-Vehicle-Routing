import numpy as np
import matplotlib.pyplot as plt
import heapq

# Grid size
N = 8

# Movement directions: (row_delta, col_delta) 
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)] # Up, down, left, right

def raw_map(prob_block, start, goal):
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

    # Randomly assign 0 (open) or 1 (blocked) for each cell
    grid = np.random.choice([0, 1], size=(N, N), p=[1 - prob_block, prob_block])

    # Guarantee that start and goal cells are open, never blocked
    # Decrease the chance to have no completed path
    grid[start[0], start[1]] = 0
    grid[goal[0], goal[1]] = 0

    # Guarantee at least one open neighbor for the start position
    start_neighbors = []
    for dx, dy in DIRS:
        nx, ny = start[0] + dx, start[1] + dy
        if 0 <= nx < N and 0 <= ny < N:
            start_neighbors.append((nx, ny))
    np.random.shuffle(start_neighbors)  # Randomize neighbor order
    neighbor_opened = any(grid[nx, ny] == 0 for nx, ny in start_neighbors)
    if not neighbor_opened and start_neighbors:
        # If no open neighbor, forcibly open the first neighbor
        nx, ny = start_neighbors[0]
        grid[nx, ny] = 0

    # Guarantee at least one open neighbor for the goal position
    goal_neighbors = []
    for dx, dy in DIRS:
        nx, ny = goal[0] + dx, goal[1] + dy
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
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def a_star(grid, start, goal, return_costs=False):
    """
    Implements the A* (A-star) search algorithm for grid-based pathfinding.

    - Uses a priority queue (heap) to always expand the cell with the lowest
      estimated total cost f = g + h.
    - Each move has a cost of 1 (assuming a uniform-cost grid).
    - Returns the shortest path if one exists, along with (optionally)
      all costs for expanded nodes.

    Parameters:
        grid : np.ndarray 2D array with 0=open, 1=obstacle.
        start : tuple (row, col), starting location.
        goal : tuple (row, col), goal location.        
        return_costs (bool): If True, return full g/h/f cost dict for all 
                             visited cells.

    Returns:
        path : list of tuple
            List of (row, col) tuples for the shortest path from start to goal,
            or None if no path exists.
        costs : dict, optional
            If return_costs=True, a dict mapping (row, col) to (g, h, f)
            for all nodes ever expanded by the algorithm.
    """
    # open_set: priority queue of nodes to explore; each entry is
    # (f, g, node, path_so_far)
    open_set = []
    heapq.heappush(open_set, (heuristic(start, goal), 0, start, [start]))

    # closed_set: set of already-expanded nodes to avoid re-exploration
    closed_set = set()

    # costs: maps each node to a tuple (g, h, f), where
    #   g = actual cost from start
    #   h = heuristic to goal
    #   f = g + h
    costs = {start: (0, heuristic(start, goal), heuristic(start, goal))}

    while open_set:
        # Pop the node with the lowest estimated total cost f
        est_total, cost, node, path = heapq.heappop(open_set)

        # If we've reached the goal, reconstruct and return the path (+costs)
        if node == goal:
            if return_costs:
                return path, costs
            return path

        if node in closed_set:
            # Don't revisit nodes
            continue

        closed_set.add(node)

        # For each possible move (up, down, left, right)
        for d in DIRS:
            nx, ny = node[0] + d[0], node[1] + d[1]
            # Make sure the neighbor is in bounds and open
            if 0 <= nx < N and 0 <= ny < N and grid[nx, ny] == 0:
                next_node = (nx, ny)
                if next_node in closed_set:
                    continue
                g = cost + 1  # Uniform cost for each move
                h = heuristic(next_node, goal)
                f = g + h
                # Only add to open_set if this path is new or strictly better
                if next_node not in costs or g < costs[next_node][0]:
                    costs[next_node] = (g, h, f)
                    heapq.heappush(open_set, (f, g, next_node, path + [next_node]))

    # No path found
    if return_costs:
        return None, costs
    return None


def plot_map_with_costs(grid, costs, path=None, title="", start=None, goal=None):
    """
    Visualizes the grid map and overlays the g/h/f values for all expanded nodes.

    - Obstacles shown as black squares, open cells as white.
    - All expanded cells (i.e., all keys in costs) have their g, h, and f values
      printed in the cell.
    - If a solution path is provided, it is drawn in red.
    - Start and goal are shown with special star markers.

    Parameters:
        grid: np.ndarray
              2D obstacle grid (0=open, 1=obstacle).
        costs: dict
               Mapping from (row, col) to (g, h, f) for all expanded cells.
        path: list of tuple, optional
              List of (row, col) tuples indicating the final solution path (if found).
        title: str, optional
               Title to display above the plot.
        start: tuple, optional
               (row, col) of the start node (displayed as green star).
        goal: tuple, optional
              (row, col) of the goal node (displayed as blue star).   

    Returns:
        None
    """
    plt.figure(figsize=(7, 7))
    plt.title(f"{title}\n(g: steps from start, h: heuristic to goal, f: g+h)")
    plt.imshow(grid, cmap='gray_r')
    plt.xticks(np.arange(N))
    plt.yticks(np.arange(N))
    plt.grid(True, color='lightgray')

    # Overlay g/h/f for all expanded cells
    for (x, y), (g, h, f) in costs.items():
        txt = f"g={g}\nh={h}\nf={f}"
        plt.text(y, x, txt, ha='center', va='center', fontsize=8,
                 color='black', 
                 bbox=dict(facecolor='white', alpha=0.65, edgecolor='none', 
                           boxstyle='round,pad=0.18'))

    # Draw the solution path in red
    if path:
        px, py = zip(*path)
        plt.plot(py, px, color='red', linewidth=3, marker='o', markersize=8, 
                 label='Path')

    # Show start and goal locations with colored stars
    if start is not None:
        plt.scatter([start[1]], [start[0]], color='green', s=150, marker='*', 
                    label='Start')
    if goal is not None:
        plt.scatter([goal[1]], [goal[0]], color='blue', s=150, marker='*', 
                    label='Goal')

    plt.legend(["Path", "Start", "Goal"],
               loc='upper left',
               bbox_to_anchor=(1.02, 1),
               borderaxespad=0)
    plt.tight_layout()
    plt.show()


def print_path_costs_and_heuristics(path, goal):
    """
    Print the actual cost (g), heuristic value (h), and f = g + h for each cell 
    on the path.

    - g: actual steps from the start to the cell (increases by 1 each move)
    - h: Manhattan distance from this cell to the goal
    - f: sum of g and h

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
        g = idx  # Each move increments g by 1
        h = heuristic(cell, goal)
        f = g + h
        print(f"  {cell}: g={g}, h={h}, f={f}")
    print()


def print_grid_stats(grid):
    """
    Prints basic statistics about the generated grid.

    - Total number of cells
    - Number and percentage of obstacles (blocked cells)
    - Number and percentage of free (open) cells

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


def create_map(prob_block, start, goal, max_attempts=1000):
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
    for attempt in range(max_attempts):
        grid = raw_map(prob_block, start, goal)
        path, costs = a_star(grid, start, goal, return_costs=True)
        if path:
            return grid, path, costs
    raise RuntimeError(
        f"Could not create a solvable map in {max_attempts} attempts." 
        "Try lowering obstacle rate or increasing attempts.")


def main():
    """
    Main function to orchestrate map generation, pathfinding, and visualization.

    - Defines the start and goal positions for all tests.
    - Runs several scenarios, each with a different obstacle probability:
      * Generates a solvable map with the specified obstacle rate.
      * Runs A* to find a path.
      * Prints map stats, the found path, and detailed cost breakdowns.
      * Visualizes the map, the solution path, and expanded node costs.

    Returns:
        None
    """
    # Set the start and goal positions for all cases
    start = (0, 0)
    goal = (7, 7)

    # Obstacle probabilities to test (0% / 20% / 50%)
    probs = [0.0, 0.2, 0.5]
    titles = ["No Obstacles (0%)", "20% Obstacles", "50% Obstacles"]

    # For each obstacle probability, generate, solve, and display results
    for prob, title in zip(probs, titles):
        print(f"\n=== Case: {title} ===")
        print(f"Start: {start}   Goal: {goal}")
        try:
            grid, path, costs = create_map(prob, start, goal)
            print_grid_stats(grid)
            print("Path found:")
            print(path)
            if path:
                print_path_costs_and_heuristics(path, goal)
            plot_map_with_costs(grid, costs, path, title=title, start=start, goal=goal)
        except RuntimeError as e:
            print(str(e))


if __name__ == "__main__":
    main()

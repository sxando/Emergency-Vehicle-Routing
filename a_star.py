import numpy as np
import matplotlib.pyplot as plt
import heapq

N = 8  # Size of the grid (N x N)
DIRS = [(-1,0),(1,0),(0,-1),(0,1)]  # 4-way movement directions: up, down, left, right

def raw_map(prob_block=0.0, start=(0,0), goal=(7,7)):
    """
    Generates a random N x N grid with given probability of obstacles,
    and guarantees that the start and goal positions are open, as well as at least one open neighbor each.
    Obstacles are represented as 1, open spaces as 0.
    Returns:
        grid: 2D numpy array of shape (N, N) with obstacles and open spaces.
    """
    grid = np.random.choice([0, 1], size=(N, N), p=[1-prob_block, prob_block])
    grid[start[0], start[1]] = 0
    grid[goal[0], goal[1]] = 0

    # Guarantee at least one open neighbor for start
    start_neighbors = []
    for dx, dy in DIRS:
        nx, ny = start[0]+dx, start[1]+dy
        if 0 <= nx < N and 0 <= ny < N:
            start_neighbors.append((nx, ny))
    np.random.shuffle(start_neighbors)  # Randomize neighbor check order
    neighbor_opened = any(grid[nx, ny] == 0 for nx, ny in start_neighbors)
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
    Compute the Manhattan distance heuristic between two points a and b.
    a, b: Tuples of (row, col)
    Returns:
        Manhattan distance: |x1-x2| + |y1-y2|
    """
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def a_star(grid, start, goal):
    """
    A* pathfinding algorithm for a 2D grid with obstacles.
    grid: 2D numpy array, 0=open, 1=blocked
    start: (row, col) tuple, starting cell
    goal: (row, col) tuple, goal cell
    Returns:
        path: List of (row, col) tuples from start to goal (inclusive), or None if no path exists

    Additional info:    
    g(n): Cost from start to n (actual cost) - Tracked during search      
    h(n): Heuristic from n to goal (estimate) - Manhattan distance to goal 
    f(n): Total score for A*(g(n) + h(n)) - Used for priority 
    """
    open_set = []
    # Heap entries: (estimated_total_cost, cost_so_far, current_cell, path_so_far)
    heapq.heappush(open_set, (heuristic(start, goal), 0, start, [start]))
    closed_set = set()
    while open_set:
        est_total, cost, node, path = heapq.heappop(open_set)
        if node == goal:
            return path
        if node in closed_set:
            continue
        closed_set.add(node)
        for d in DIRS:
            nx, ny = node[0]+d[0], node[1]+d[1]
            # Stay in bounds and only expand open cells
            if 0<=nx<N and 0<=ny<N and grid[nx,ny]==0:
                next_node = (nx, ny)
                if next_node in closed_set:
                    continue
                heapq.heappush(open_set, (cost+1+heuristic(next_node, goal), cost+1, next_node, path+[next_node]))
    return None  # No path found

def plot_map(grid, path=None, title="", start=(0,0), goal=(7,7)):
    """
    Visualize the grid, the found path, and highlight start and goal.
    - Obstacles are black, open cells are white.
    - Path is shown in red.
    - Start is a green star, goal is a blue star.
    """
    plt.figure(figsize=(5,5))
    plt.imshow(grid, cmap='gray_r')
    plt.xticks(np.arange(N))
    plt.yticks(np.arange(N))
    plt.grid(True)
    plt.title(title)
    # Draw path if present
    if path:
        px, py = zip(*path)
        plt.plot(py, px, color='red', linewidth=3, marker='o', markersize=8)
    # Draw start and goal
    plt.scatter([start[1]], [start[0]], color='green', s=150, marker='*', label='Start')
    plt.scatter([goal[1]], [goal[0]], color='blue', s=150, marker='*', label='Goal')
    plt.legend(["Path", "Start", "Goal"])
    plt.show()

def create_map(prob_block=0.0, max_attempts=1000, start=(0,0), goal=(7,7)):
    """
    Generates a random map and guarantees that there is a valid path between start and goal.
    Tries up to max_attempts times; raises an error if no solvable map is found.
    Returns:
        grid: 2D numpy array
        path: List of (row, col) tuples representing the solution path
    """
    for attempt in range(max_attempts):
        grid = raw_map(prob_block=prob_block, start=start, goal=goal)
        path = a_star(grid, start, goal)
        if path:
            return grid, path
    raise RuntimeError(f"Could not create a solvable map in {max_attempts} attempts. Try lowering obstacle rate or increasing attempts.")

def print_heuristic_grid(goal, N=8):
    """
    Prints the full Manhattan distance heuristic grid for the goal point.
    Each cell's value shows its heuristic estimate to the goal.
    """
    print(f"\nHeuristic (Manhattan distance) grid to goal {goal}:")
    header = "    " + " ".join([f"{j:2d}" for j in range(N)])
    print(header)
    print("   " + "---"*N)
    for i in range(N):
        row = [f"{heuristic((i,j), goal):2d}" for j in range(N)]
        print(f"{i:2d}| " + " ".join(row))
    print()

def print_path_heuristics(path, goal):
    """
    Prints the heuristic value for each cell along the found path.
    """
    print("Heuristic values along the path:")
    for cell in path:
        h = heuristic(cell, goal)
        print(f"  {cell}: h={h}")
    print()

def print_grid_stats(grid):
    """
    Prints statistics about the generated grid:
      - Total number of cells
      - Number and percentage of obstacles
      - Number and percentage of free cells
    """
    total = grid.size
    obstacles = np.sum(grid == 1)
    free = np.sum(grid == 0)
    percent_obstacles = 100.0 * obstacles / total
    percent_free = 100.0 * free / total
    print(f"Total cells: {total}")
    print(f"Obstacles: {obstacles} ({percent_obstacles:.1f}%)")
    print(f"Free cells: {free} ({percent_free:.1f}%)\n")

def main():
    """
    Main function that runs the map generation, A* algorithm, and visualization
    for several obstacle probabilities.
    """
    # Define the start and goal points
    start = (0,0)
    goal = (7,7)

    # Set the probabilities for obstacle appearance
    probs = [0.0, 0.2, 0.5]
    titles = ["No Obstacles (0%)", "20% Obstacles", "50% Obstacles"]

    # For each obstacle probability, create a map, run A*, and display/print everything
    for prob, title in zip(probs, titles):
        print(f"\n=== Case: {title} ===")
        print(f"Start: {start}   Goal: {goal}")
        grid, path = create_map(prob_block=prob, start=start, goal=goal)
        print_grid_stats(grid)
        print_heuristic_grid(goal, N)
        print("Path found:")
        print(path)
        if path:
            print_path_heuristics(path, goal)
        plot_map(grid, path, title=title, start=start, goal=goal)

if __name__ == "__main__":
    main()

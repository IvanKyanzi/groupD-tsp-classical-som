"""
tsp_dp.py: Dynamic Programming (DP) solution for the Traveling Salesman Problem (TSP).
- Solves TSP for 7 cities using DP.
- Route must start and end at City 1, using only direct edges.
"""

# Task 1: Graph Representation
# adjacency matrix with direct edge between City 6 and City 4 (assumed distance 10)
INF = float('inf')
adj_matrix = [
    [0, 12, 10, INF, INF, INF, 12],  # City 1
    [12, 0, 8, 12, INF, INF, INF],   # City 2
    [10, 8, 0, 11, 3, INF, INF],     # City 3
    [INF, 12, 11, 0, 11, 10, INF],   # City 4 (4->6: 10)
    [INF, INF, 3, 11, 0, 6, 7],      # City 5
    [INF, INF, INF, 10, 6, 0, 9],    # City 6 (6->4: 10)
    [12, INF, INF, INF, 7, 9, 0]      # City 7
]

# Task 2: Classical TSP Solution (Dynamic Programming)
def tsp_dp(adj_matrix, start):
    """
    Solves TSP using Dynamic Programming.
    Args:
        adj_matrix (list of lists): Adjacency matrix of the graph.
        start (int): Starting city index (0 for City 1).
    Returns:
        path (list): Optimal route (0-based indices).
        min_cost (float): Total distance of the route.
    """
    n = len(adj_matrix)
    dp = {}
    parent = {}
    
    # Initialize DP table
    for mask in range(1 << n):
        for v in range(n):
            dp[(mask, v)] = INF
            parent[(mask, v)] = -1
    dp[(1 << start, start)] = 0
    
    # Fill DP table
    for mask in range(1 << n):
        for u in range(n):
            if not (mask & (1 << u)):
                continue
            for v in range(n):
                if mask & (1 << v):
                    continue
                if adj_matrix[u][v] == INF:
                    continue
                new_mask = mask | (1 << v)
                cost = dp[(mask, u)] + adj_matrix[u][v]
                if cost < dp[(new_mask, v)]:
                    dp[(new_mask, v)] = cost
                    parent[(new_mask, v)] = u
    
    # Find minimum cost to complete the tour
    final_mask = (1 << n) - 1
    min_cost = INF
    last_city = -1
    for u in range(n):
        if u == start:
            continue
        if adj_matrix[u][start] == INF:
            continue
        cost = dp[(final_mask, u)] + adj_matrix[u][start]
        if cost < min_cost:
            min_cost = cost
            last_city = u
    
    if min_cost == INF:
        return None, None
    
    # Reconstruct the path
    path = []
    mask = final_mask
    current = last_city
    while current != -1:
        path.append(current)
        next_mask = mask ^ (1 << current)
        current = parent[(mask, current)]
        mask = next_mask
    path = path[::-1]
    path.append(start)
    
    return path, min_cost

# Main Execution
def main():
    print("=== TSP Assignment: Dynamic Programming (DP) Solution ===")
    
    # Run DP
    print("\nClassical TSP Solution (Dynamic Programming)")
    dp_path, dp_distance = tsp_dp(adj_matrix, 0)
    if dp_path:
        dp_path = [city + 1 for city in dp_path]  # Convert to 1-based indices
        print("DP Route:", dp_path)
        print("Total Distance:", dp_distance)
    else:
        print("No valid path exists using DP.")

if __name__ == "__main__":
    main()
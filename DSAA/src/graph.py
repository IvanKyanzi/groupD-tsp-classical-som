"""
Graph representation for the TSP assignment.
Defines the adjacency matrix for the 7 cities.
"""

INF = float('inf')

#  adjacency matrix with direct edge between City 6 and City 4 
adj_matrix = [
    [0, 12, 10, INF, INF, INF, 12],  # City 1
    [12, 0, 8, 12, INF, INF, INF],   # City 2
    [10, 8, 0, 11, 3, INF, INF],     # City 3
    [INF, 12, 11, 0, 11, 10, INF],   # City 4 (4->6: 10)
    [INF, INF, 3, 11, 0, 6, 7],      # City 5
    [INF, INF, INF, 10, 6, 0, 9],    # City 6 (6->4: 10)
    [12, INF, INF, INF, 7, 9, 0]      # City 7
]
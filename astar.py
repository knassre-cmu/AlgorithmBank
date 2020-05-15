from graph import *

'''
AStar

Inputs

graph: an instance of the Graph class
start: a node in graph
target: a node in graph
heuristic: a function which takes in a node and the target node and returns a 
heuristic value (use lambda x,y: 0 to turn it from AStar to Dijkstra)

Outputs

path: a list of nodes connecting start to target with the minimum weight.
Returns None if there is no path.
'''

# Wrapper function for Astar. Sets up the unvisited set, distance dict, 
# path dict and neighbor dict.
def ASTAR(graph,start,target,heuristic):
    unvisited = set(graph.nodes)
    distances = {n:float("inf") for n in graph.nodes}
    distances[start] = 0
    paths = {start:[start]}
    neighbors = graph.neighborDict()
    return ASTARHELPER(graph,target,unvisited,distances,paths,neighbors,heuristic)

# Main function for AStar. Finds the unvisited node with the smallest distance
# to start node + heuristic distance, marks it as visited, finds the distance of its
# neighbors and updates their paths. Stops whent the target is reached.
def ASTARHELPER(graph,target,unvisited,distances,paths,neighbors,heuristic):
    if unvisited == set(): return
    node = min(unvisited,key = lambda x: distances[x] + heuristic(x,target))
    unvisited.discard(node)
    if node == target: return paths[node]
    for neighbor in neighbors[node]:
        weight = graph.edges[node][neighbor] + distances[node]
        if weight < distances[neighbor]:
            distances[neighbor] = weight
            paths[neighbor] = paths[node] + [neighbor]
    return ASTARHELPER(graph,target,unvisited,distances,paths,neighbors,heuristic)

# Test cases for ASTAR
def TESTASTAR():
    print("Testing AStar:")
    G1 = GRAPH()
    G2 = GRAPH()
    G3 = GRAPH()
    G4 = GRAPH()
    for i in range(1,4): G1.addNode(i)
    for i in range(1,5): G2.addNode(i)
    for i in range(1,7): G3.addNode(i)
    for i in range(1,8): G4.addNode(i)
    E1 = [(2,1),(1,2),(3,1),(1,3),(3,2),(2,3)]
    E2 = [(3,4),(1,2),(1,3),(4,1),(4,2),(2,4),(3,2),(2,3),(2,1),(1,4),(4,3),(3,1)]
    E3 = [(1,4),(4,2),(4,5),(6,3),(3,5),(4,6),(5,6),(3,2),(3,6),(6,5),(5,2),(2,4),
          (3,1),(5,3),(4,3),(2,6),(5,4),(4,1),(5,1),(2,3)]
    E4 = [(7,4),(6,2),(4,7),(4,6),(2,3),(4,1),(6,7),(4,5),(5,1),(1,4),(7,1),(2,5),
          (5,2),(1,6),(1,3),(3,7),(1,7),(5,7),(2,6),(6,3)]
    for i in range(len(E1)): G1.addEdge(E1[i][0],E1[i][1],len(E1)-i)
    for i in range(len(E2)): G2.addEdge(E2[i][0],E2[i][1],len(E2)-i)
    for i in range(len(E3)): G3.addEdge(E3[i][0],E3[i][1],len(E3)-i)
    for i in range(len(E4)): G4.addEdge(E4[i][0],E4[i][1],len(E4)-i)
    h = lambda x,y:abs(y-x)
    print(chr(10209)+" Test Case 1:","PASSED" if ASTAR(G1,3,1,h) == [3,1] else "FAILED")
    print(chr(10209)+" Test Case 2:","PASSED" if ASTAR(G2,4,1,h) == [4,1] else "FAILED")
    print(chr(10209)+" Test Case 3:","PASSED" if ASTAR(G3,6,1,h) == [6,2,3,1] else "FAILED")
    print(chr(10209)+" Test Case 4:","PASSED" if ASTAR(G4,7,2,h) == [7,3,6,2] else "FAILED")
    print()

if __name__ == "__main__": TESTASTAR()
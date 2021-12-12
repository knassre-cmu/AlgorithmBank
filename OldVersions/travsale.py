from graph import *
from hillclimb import *
import copy

'''
Traveling Salesman Hillclimber

Inputs

graph: an instance of the class Graph where all nodes are connected
n: a number of times to iterate the hill climbing algorithm

Outputs

path: a list of all nodes in graph ordered to optimize the travel time
'''

# Calls HILLCLIMBER to heuristically optimize the best route between all the nodes
def TRAVSALE(graph,n):
    initial = list(graph.nodes)
    return HILLCLIMBER(initial,lambda p: TRAVLEN(p,graph),TRAVSWAP,n)

# Returns the length of a certain path between all the nodes
def TRAVLEN(path,graph):
    length = 0
    for i in range(len(path)-1):
        length += graph.edges[path[i]][path[i+1]]
    return length

# Returns all orderings of the nodes that can be achieved from a single swap
def TRAVSWAP(path):
    out = []
    for i in range(len(path)):
        for j in range(i,len(path)):
            swapped = copy.deepcopy(path)
            swapped[i], swapped[j] = swapped[j], swapped[i]
            out.append(swapped)
    if path in out: out.remove(path)
    return out

# Test cases for TRAVSALE
def TESTTRAVSALE():
    print("Testing Traveling Salemsan:")
    G1 = GRAPH()
    G2 = GRAPH()
    G3 = GRAPH()
    G4 = GRAPH()
    N1 = [(1,1),(1,3),(5,0),(4,2),(2,0),(3,1),(4,5),(6,3)]
    N2 = [(10,10),(11,20),(12,15),(13,14),(14,17),(15,11),(16,19),(17,18),(18,13),(19,16),(20,12)]
    N3 = [(36,66),(34,41),(96,56),(92,81),(60,78),(44,54),(87,16),(80,51),(71,85),(66,91),(20,89),
          (20,69),(27,84),(25,64),(2,48),(39,21)]
    for n in N1: G1.addNode(n)
    for i in range(len(N1)):
        for j in range(i+1,len(N1)):
            dX = N1[i][0]-N1[j][0]
            dY = N1[i][1]-N1[j][1]
            G1.addEdge(N1[i],N1[j],(dX**2+dY**2)**0.5)
    for n in N2: G2.addNode(n)
    for i in range(len(N2)):
        for j in range(i+1,len(N2)):
            dX = N2[i][0]-N2[j][0]
            dY = N2[i][1]-N2[j][1]
            G2.addEdge(N2[i],N2[j],(dX**2+dY**2)**0.5)
    for n in N3: G3.addNode(n)
    for i in range(len(N3)):
        for j in range(i+1,len(N3)):
            dX = N3[i][0]-N3[j][0]
            dY = N3[i][1]-N3[j][1]
            G3.addEdge(N3[i],N3[j],(dX**2+dY**2)**0.5)
    P1 = [(4,5),(1,3),(1,1),(2,0),(3,1),(5,0),(4,2),(6,3)]
    P2 = [(10,10),(15,11),(18,13),(20,12),(19,16),(13,14),(12,15),(14,17),(17,18),(16,19),(11,20)]
    P3 = [(92,81),(96,56),(80,51),(87,16),(39,21),(27,84),(20,89),(20,69),(2,48),(34,41),(44,54),
          (25,64),(36,66),(60,78),(71,85),(66,91)]
    print(chr(10209)+" Test Case 1:","PASSED" if TRAVSALE(G1,5) == P1 else "FAILED")
    print(chr(10209)+" Test Case 2:","PASSED" if TRAVSALE(G2,8) == P2 else "FAILED")
    print(chr(10209)+" Test Case 3:","PASSED" if TRAVSALE(G3,11) == P3 else "FAILED")
    print()

if __name__ == "__main__": TESTTRAVSALE()

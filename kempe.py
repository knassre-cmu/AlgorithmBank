from graph import *
import copy

'''
Kempe's Algorithm

Inputs

graph: an instance of the Graph class

Outputs

cDict: a dictionary where the keys are the nodes in graph and the values are
the numerical colors of each node, such that no node shares an edge with a node
of the same color and ideally with the minimum number of colors.
'''

# Wrapper function for Kempe's algorithm. Sets up the dummy graph.
def KEMPE(graph):
    gprime = copy.deepcopy(graph)
    return KEMPEHELPER(graph,gprime)

# Takes in a dummy graph and removes the vertices from the graph in order of
# minimum degree, updating the degree of each node along the way and pushing
# the nodes onto the stack.
def GRAPHPOPPER(gprime):
    stack = []
    degrees = {n:0 for n in gprime.nodes}
    neighbors = gprime.neighborDict()
    for node in gprime.nodes:
        for neighbor in neighbors[node]:
            degrees[node] += 1
    while gprime.nodes != set():
        minNode = min(gprime.nodes, key = lambda n: degrees[n])
        gprime.nodes.discard(minNode)
        for neighbor in neighbors[minNode]:
            degrees[neighbor] -= 1
        stack.append(minNode)
    return stack

# Main function for Kempe's algorithm. First calls the graph popper to obtain 
# the coloring stack. Then, colors in the nodes in the order they come off 
# the stack with the maximum color of its neighbors + 1
def KEMPEHELPER(graph,gprime):
    cDict = {}
    stack = GRAPHPOPPER(gprime)
    neighbors = graph.neighborDict()
    while stack != []:
        node = stack.pop()
        neighborColors = set()
        for neighbor in neighbors[node]:
            neighborColors.add(cDict.get(neighbor,0))
        i = 1
        while i in neighborColors: i += 1
        cDict[node] = i
    return cDict

# Test cases for KEMPE
def TESTKEMPE():
    print("Testing Kempe's Algorithm")
    G1 = GRAPH()
    G2 = GRAPH()
    G3 = GRAPH()
    G4 = GRAPH()
    for i in range(1,7): G1.addNode(i)
    for i in range(1,9): G2.addNode(i)
    for i in range(1,13): G3.addNode(i)
    for i in range(1,11): G4.addNode(i)
    E1 = [(1,2),(2,3),(3,4),(5,6),(6,1),(1,4),(3,5)]
    E2 = [(1,2),(1,3),(1,4),(1,5),(1,6),(1,7),(1,8),(2,3),(2,4),(3,4),(4,5),
          (5,6),(6,7),(6,8),(7,8)]
    E3 = [(1,2),(1,3),(1,4),(1,8),(1,9),(2,3),(2,4),(2,6),(2,5),(3,6),(3,7),(3,8),
          (4,5),(4,9),(4,10),(5,6),(5,10),(5,11),(6,7),(6,11),(7,8),(7,11),(7,12),
          (8,9),(8,12),(9,10),(9,12),(10,11),(10,12),(11,12)]
    E4 = [(1,2),(1,5),(1,6),(2,3),(2,7),(3,4),(3,8),(4,5),(4,9),(5,10),(6,8),(6,9),
          (7,9),(7,10),(8,10)]
    for i in range(len(E1)): G1.addEdge(E1[i][0],E1[i][1],1)
    for i in range(len(E2)): G2.addEdge(E2[i][0],E2[i][1],1)
    for i in range(len(E3)): G3.addEdge(E3[i][0],E3[i][1],1)
    for i in range(len(E4)): G4.addEdge(E4[i][0],E4[i][1],1)
    C1 = {1:3, 2:2, 3:1, 4:2, 5:2, 6:1}
    C2 = {1:4, 2:3, 3:1, 4:2, 5:1, 6:3, 7:2, 8:1}
    C3 = {12:1, 11:2, 10:3, 9:2, 8:3, 7:4, 6:1, 5:4, 4:1, 3:2, 2:3, 1:4}
    C4 = {10:1, 7:2, 9:1, 8:2, 6:3, 5:2, 4:3, 3:1, 2:3, 1:1}
    print(chr(10209)+" Test Case 1:","PASSED" if KEMPE(G1)== C1 else "FAILED")
    print(chr(10209)+" Test Case 2:","PASSED" if KEMPE(G2)== C2 else "FAILED")
    print(chr(10209)+" Test Case 3:","PASSED" if KEMPE(G3)== C3 else "FAILED")
    print(chr(10209)+" Test Case 4:","PASSED" if KEMPE(G4)== C4 else "FAILED")

if __name__ == "__main__": TESTKEMPE()
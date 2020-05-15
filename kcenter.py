from graph import *
from hillclimb import *
import copy

'''
K Centers Algorithm

Inputs

graph: an instance of the class Graph where all nodes are connected
seed: the node belonging to the first center
k: a number of centers to place within the nodes

Outputs

cSet: the set of k nodes with centers such that each node has a heuristically
minimized distance to the nearest center.
'''

# Wrapper function for K Centers algorithm. Sets up the center set and open set
def KCENTERS(graph,seed,k):
    cSet = {seed}
    KCENTERSHELPER(graph,cSet,graph.nodes-cSet,k-1)
    return cSet

# Main function for K Centers algorithm. Places a center in the node that is
# farthest from any center.
def KCENTERSHELPER(graph,cSet,oSet,k):
    if k <= 0 or oSet == set(): return
    maxNode = None
    maxDist = float("-inf")
    for node in oSet:
        minDist = float("inf")
        for center in cSet:
            dist = graph.edges[node][center]
            if dist < minDist: minDist = dist
        if minDist > maxDist:
            maxNode = node
            maxDist = minDist
    cSet.add(maxNode)
    oSet.discard(maxNode)
    KCENTERSHELPER(graph,cSet,oSet,k-1)

# Test cases for KCENTERS
def TESTKCENTERS():
    print("Testing K Centers:")
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
    C1 = {(4,5),(4,2),(1,3)}
    C2 = {(15,11),(14,17),(20,12),(10,10)}
    C3 = {(2,48),(71,85),(20,89),(87,16),(39,21),(96,56)}
    print(chr(10209)+" Test Case 1:","PASSED" if KCENTERS(G1,(4,2),3) == C1 else "FAILED")
    print(chr(10209)+" Test Case 2:","PASSED" if KCENTERS(G2,(14,17),4) == C2 else "FAILED")
    print(chr(10209)+" Test Case 3:","PASSED" if KCENTERS(G3,(71,85),6) == C3 else "FAILED")
    print()

if __name__ == "__main__": TESTKCENTERS()
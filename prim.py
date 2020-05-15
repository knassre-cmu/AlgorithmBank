from graph import *
from pq import *

'''
Prim's Algorithm

Inputs

graph: an instance of the Graph class
seed: an arbitrary node from graph to start the algorithm from

Outputs

mst: a set of some edges from the graph, such that mst represents a minimum 
spanning tree of the graph. A spanning tree is a subset of the graph such that
every node is connected and there are no cycles. A minimum spanning tree is 
a spanning tree with the minimum weight.
'''

# Wrapper function for Prim's algorithm. Initializes the visited set and the priority queue.
def PRIM(graph,seed):
    neighbors = graph.neighborDict()
    pq = PRIORITYQUEUE()
    for neighbor in neighbors[seed]:
        pq.add((seed,neighbor),graph.edges[seed][neighbor])
    mst = set()
    PRIMHELPER(pq,mst,{seed},neighbors,graph)
    return mst

# Main function of the algorithm. Removes the edge from the priority queue with the minimum
# weight and adds it to the MST if only 1 of the 2 nodes is has been visited. Updates the
# visited set and the MST, and adds all edges of the second node to the priority queue.
def PRIMHELPER(pq,mst,visited,neighborDict,graph):
    if pq.isEmpty(): return
    if len(mst) == len(neighborDict)**2-1: return
    edge, weight = pq.rem()
    node1, node2 = edge
    if not (node1 in visited and node2 in visited):
        mst.add(edge)
        visited.add(node2)
        for neighbor in neighborDict[node2]:
            pq.add((node2,neighbor),graph.edges[node2][neighbor])
    PRIMHELPER(pq,mst,visited,neighborDict,graph)

# Checks if 2 sets of edges are equal
def ESEQ(ST1,ST2):
    d1 = set()
    d2 = set()
    for e in ST1:
        d1.add(e)
        d1.add(e[::-1])
    for e in ST2:
        d2.add(e)
        d2.add(e[::-1])
    return d1 == d2

# Test cases for PRIM
def TESTPRIM():
    print("Testing Prim's Algorithm:")
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
    MST1 = {(1,3),(2,3)}
    MST2 = {(3,1),(2,1),(4,3)}
    MST3 = {(2,6),(2,3),(4,3),(5,1),(4,1)}
    MST4 = {(2,6),(5,7),(1,4),(6,3),(1,7),(3,7)}
    print(chr(10209)+" Test Case 1:","PASSED" if ESEQ(PRIM(G1,1),MST1) else "FAILED")
    print(chr(10209)+" Test Case 2:","PASSED" if ESEQ(PRIM(G2,1),MST2) else "FAILED")
    print(chr(10209)+" Test Case 3:","PASSED" if ESEQ(PRIM(G3,1),MST3) else "FAILED")
    print(chr(10209)+" Test Case 4:","PASSED" if ESEQ(PRIM(G4,1),MST4) else "FAILED")
    print()

if __name__ == '__main__': TESTPRIM()
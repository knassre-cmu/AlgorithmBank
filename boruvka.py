from graph import *

'''
Borůvka's Algorithm

Inputs

graph: an instance of the graph class

Outputs:

mst: a set of some edges from the graph, such that mst represents a minimum 
spanning tree of the graph. A spanning tree is a subset of the graph such that
every node is connected and there are no cycles.A minimum spanning tree is 
a spanning tree with the minimum weight.
'''

# Wrapper function for Borůvka's algorithm. Sets up the union find dict.
def BORUVKA(graph):
    ufDict = {n:-1 for n in graph.nodes}
    mst = set()
    BORUVKAHELPER(graph,mst,ufDict)
    return mst

# Finds the canonical representative of a node using height tracking and path compression
def FIND(node,ufDict):
    parent = ufDict[node]
    if isinstance(parent,int) and parent < 1: return node
    rep = FIND(parent,ufDict)
    ufDict[node] = rep
    return rep

# Merges two canonical representatives using height tracking
def UNION(node1,node2,ufDict):
    height1 = ufDict[node1]
    height2 = ufDict[node2]
    if height1 < height2: ufDict[node2] = node1
    else: ufDict[node1] = node2
    if height1 == height2: ufDict[node2] -= 1

# Main function for Borůvka's algorithm. Finds all distinct forests in the graph,
# loops through each edge that connects 2 distinct forests and checks if it is the
# cheapest edge connecting to that forest in that round. At the end of the round,
# adds all cheapest edges to the forest and updates the union find dictionary.
def BORUVKAHELPER(graph,mst,ufDict):
    trees = {FIND(n,ufDict) for n in ufDict}
    if len(trees) == 1: return
    cheapest = {t:None for t in trees}
    for edge in graph.edges:
        node1, node2 = edge
        tree1, tree2 = FIND(node1,ufDict), FIND(node2,ufDict)
        if tree1 != tree2:
            if graph.edges[edge] < graph.edges.get(cheapest[tree1],float("inf")):
                cheapest[tree1] = edge
            if graph.edges[edge] < graph.edges.get(cheapest[tree2],float("inf")):
                cheapest[tree2] = edge
    for edge in list(cheapest.values()):
        if edge != None: 
            mst.add(edge)
            tree1, tree2 = FIND(edge[0],ufDict), FIND(edge[1],ufDict)
            if tree1 != tree2: UNION(tree1,tree2,ufDict)
    BORUVKAHELPER(graph,mst,ufDict)

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

# Test cases for BORUVKA
def TESTBORUVKA():
    print("Testing Borůvka's Algorithm:")
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
    MST2 = {(1,2),(1,3),(3,4)}
    MST3 = {(1,4),(1,5),(2,3),(2,6),(3,4)}
    MST4 = {(2,6),(5,7),(1,4),(6,3),(1,7),(3,7)}
    print(chr(10209)+" Test Case 1:","PASSED" if ESEQ(BORUVKA(G1),MST1) else "FAILED")
    print(chr(10209)+" Test Case 2:","PASSED" if ESEQ(BORUVKA(G2),MST2) else "FAILED")
    print(chr(10209)+" Test Case 3:","PASSED" if ESEQ(BORUVKA(G3),MST3) else "FAILED")
    print(chr(10209)+" Test Case 4:","PASSED" if ESEQ(BORUVKA(G4),MST4) else "FAILED")
    print()

if __name__ == "__main__": TESTBORUVKA()
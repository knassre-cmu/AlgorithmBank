from graph import *

'''
Depth-First Search

A - DFS Path

Inputs

graph: an instance of the graph class
start: a node in graph
target: a node in graph

Outputs

path: a list of nodes connecting start to target. Returns None if there is no path.

B - DFS Span

graph: an instance of the graph class
seed: an arbitrary node from graph to start the algorithm from

Outputs

st: a set of some edges from the graph, such that mst represents a minimum 
spanning tree of the graph. A spanning tree is a subset of the graph such that
every node is connected and there are no cycles.
'''

# Wrapper function for DFS Path. Sets up the visited set and neighbor dict.
def DFSPATH(graph,start,target):
    return DFSPATHHELPER(graph,start,target,set(),graph.neighborDict())

# Main function for DFS Path. Loops through each neighbor, making recursive calls
# and returning the path + the current node if the recursive call bears fruit.
def DFSPATHHELPER(graph,start,target,visited,neighbors):
    if start == target: return [target]
    if start in visited: return None
    visited.add(start)
    for neighbor in neighbors[start]:
        path = DFSPATHHELPER(graph,neighbor,target,visited,neighbors)
        if path != None: return [start] + path
    visited.discard(start)
    return None

# Wrapper function for DFS Span. Sets up the visited set and neighbor dict
def DFSSPAN(graph,seed):
    st = set()
    DFSSPANHELPER(graph,st,seed,set(),graph.neighborDict())
    return st

# Main function for DFS Span. Loops through each neighbor, making recursive calls
# and adding edges to the spanning tree until every node is visited.
def DFSSPANHELPER(graph,st,seed,visited,neighbors):
    if seed in visited: return
    visited.add(seed)
    for neighbor in neighbors[seed]:
        if neighbor not in visited:
            st.add((seed,neighbor))
            DFSSPANHELPER(graph,st,neighbor,visited,neighbors)

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

# Test cases for DFSPATH and DFSSPAN
def TESTDFS():
    print("Testing DFS Path:")
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
    ST1 = {(1,2),(2,3)}
    ST2 = {(1,2),(3,4),(2,3)}
    ST3 = {(1,3),(3,2),(2,4),(4,5),(5,6)}
    ST4 = {(1,3),(3,2),(2,5),(5,4),(4,6),(6,7)}
    print(chr(10209)+" Test Case 1:","PASSED" if DFSPATH(G1,1,3) == [1,2,3] else "FAILED")
    print(chr(10209)+" Test Case 2:","PASSED" if DFSPATH(G2,1,4) == [1,2,3,4] else "FAILED")
    print(chr(10209)+" Test Case 3:","PASSED" if DFSPATH(G3,1,6) == [1,3,2,4,5,6] else "FAILED")
    print(chr(10209)+" Test Case 4:","PASSED" if DFSPATH(G4,3,7) == [3,1,4,5,2,6,7] else "FAILED")
    print("Testing DFS Span:")
    print()
    print(chr(10209)+" Test Case 1:","PASSED" if ESEQ(DFSSPAN(G1,1),ST1) else "FAILED")
    print(chr(10209)+" Test Case 2:","PASSED" if ESEQ(DFSSPAN(G2,1),ST2) else "FAILED")
    print(chr(10209)+" Test Case 3:","PASSED" if ESEQ(DFSSPAN(G3,1),ST3) else "FAILED")
    print(chr(10209)+" Test Case 4:","PASSED" if ESEQ(DFSSPAN(G4,1),ST4) else "FAILED")
    print()

if __name__ == "__main__": TESTDFS()

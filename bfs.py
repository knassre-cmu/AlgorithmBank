from graph import *
from queue import *

'''
Bredth-First Search

A - BFS Path

Inputs

graph: an instance of the graph class
start: a node in graph
target: a node in graph

Outputs

path: a list of nodes connecting start to target. Returns None if there is no path.

B - BFS Span

graph: an instance of the graph class
seed: an arbitrary node from graph to start the algorithm from

Outputs

st: a set of some edges from the graph, such that mst represents a minimum 
spanning tree of the graph. A spanning tree is a subset of the graph such that
every node is connected and there are no cycles.
'''

# Wrapper function for BFS Path. Sets up the worklist, visited set and neighbor dict 
def BFSPATH(graph,start,target):
    queue = QUEUE()
    queue.enq([start])
    return BFSPATHHELPER(graph,queue,target,{start},graph.neighborDict())

# Main function for BFS Path. Takes the first path from the worklist, checks if it
# extends to the target node, and if not enqueues the path + every unvisited neighbor
# of the last item in the path, before making a recursive call.
def BFSPATHHELPER(graph,queue,target,visited,neighbors):
    if queue == []: return None
    path = queue.deq()
    last = path[-1] 
    if last == target: return path
    for neighbor in neighbors[last]:
        if neighbor in visited: continue
        visited.add(neighbor)
        queue.enq(path+[neighbor])
    return BFSPATHHELPER(graph,queue,target,visited,neighbors)

# Wrapper function for BFS Span. Sets up the worklist, visited set and neighbor dict 
def BFSSPAN(graph,seed):
    neighbors = graph.neighborDict()
    queue = QUEUE()
    for neighbor in neighbors[seed]:
        queue.enq((seed,neighbor))
    st = set()
    BFSSPANHELPER(graph,st,queue,{seed},neighbors)
    return st

# Main function for BFS Span. Takes the first edge from the worklist, loops
# through all of the unvisited neighbors neighbors of the nodes, adds them to the MST,
# then makes a recursive call
def BFSSPANHELPER(graph,st,queue,visited,neighbors):
    if queue == []: return
    if len(st) == len(graph.nodes)-1: return
    edge = queue.deq()
    node1, node2 = edge
    for neighbor in neighbors[node2]:
        if neighbor in visited: continue
        visited.add(neighbor)
        queue.enq((node2,neighbor))
    st.add(edge)
    BFSSPANHELPER(graph,st,queue,visited,neighbors)

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

# Test cases for BFSPATH and BFSSPAN
def TESTBFS():
    print("Testing BFS Path:")
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
    ST1 = {(1,2),(1,3)}
    ST2 = {(1,2),(1,3),(1,4)}
    ST3 = {(1,3),(1,4),(1,5),(2,3),(3,4)}
    ST4 = {(1,3),(1,4),(1,5),(1,6),(1,7),(2,3)}
    print(chr(10209)+" Test Case 1:","PASSED" if BFSPATH(G1,1,3) == [1,3] else "FAILED")
    print(chr(10209)+" Test Case 2:","PASSED" if BFSPATH(G2,1,4) == [1,4] else "FAILED")
    print(chr(10209)+" Test Case 3:","PASSED" if BFSPATH(G3,1,6) == [1,3,6] else "FAILED")
    print(chr(10209)+" Test Case 4:","PASSED" if BFSPATH(G4,3,7) == [3,7] else "FAILED")
    print()
    print("Testing BFS Span:")
    print(chr(10209)+" Test Case 1:","PASSED" if ESEQ(BFSSPAN(G1,1),ST1) else "FAILED")
    print(chr(10209)+" Test Case 2:","PASSED" if ESEQ(BFSSPAN(G2,1),ST2) else "FAILED")
    print(chr(10209)+" Test Case 3:","PASSED" if ESEQ(BFSSPAN(G3,1),ST3) else "FAILED")
    print(chr(10209)+" Test Case 4:","PASSED" if ESEQ(BFSSPAN(G4,1),ST4) else "FAILED")
    print()

if __name__ == "__main__": TESTBFS()
'''
GRAPH Class

Represents an abstract weighted & undirected graph using a set of nodes and
a dictionary of edges, where the keys are nodes, and the values are a dictionary
of connected nodes where the keys are nodes and the value are weights of the
connection. Use .addNode() to add a new node and .addEdge() to connect 2 nodes. 
Use .neighborDict() to get a dictionary of all nodes connected to each node. 
Use .deleteNode() to remove a node and all corresponding edges from the graph. 
Use .hasEdge() to detect if 2 nodes are connected via an edge. Use .getWeight()
to return the weight of an edge between two connected nodes. Graphs can be printed
and will display all nodes and their neighbors, with weghts
'''

class GRAPH(object):
    def __init__(self):
        self.nodes = set()
        self.edges = dict()

    def __repr__(self):
        arr = []
        for node in self.nodes:
            s = f"{node} => "
            edges = []
            for neighbor in self.edges[node]:
                edges.append(f"({neighbor},{self.edges[node][neighbor]})")
            arr.append(s+" ".join(edges))
        return "\n".join(arr)

    # Adds node to the graph
    def addNode(self,node):
        self.nodes.add(node)
        self.edges[node] = dict()

    # Adds an edge between node1 and node2 with a weight of weight
    def addEdge(self,node1,node2,weight):
        self.edges[node1][node2] = weight
        self.edges[node2][node1] = weight

    def hasEdge(self,node1,node2):
        return node2 in self.edges[node1]

    def getWeight(self,node1,node2):
        return self.edges[node1][node2]

    # Removes a node from the graph and all corresponding edges.
    def deleteNode(self,node):
        self.nodes.discard(node)
        del self.edges[node]
        for other in self.edges:
            if node in self.edges[other]:
                del self.edges[other][node] 

    # Removes an edge from the node
    def deleteEdge(self,node1,node2):
        del self.edges[node1][node2]
        del self.edges[node2][node1]

    # Returns a dictionary of all nodes and the set of their neighbors
    def neighborDict(self):
        nDict = {}
        for node in self.nodes:
            nDict[node] = set()
        for node1 in self.edges:
            for node2 in self.edges[node1]:
                nDict[node1].add(node2)
                nDict[node2].add(node1)
        return nDict

# Test cases for GRAPH Class
def TESTGRAPH():
    print("Testing GRAPH Class:")
    G1 = GRAPH()
    G2 = GRAPH()
    G3 = GRAPH()
    for i in range(1,4): G1.addNode(i)
    for i in range(1,5): G2.addNode(i)
    for i in range(1,7): G3.addNode(i)
    E1 = [(2,1),(1,2),(3,1),(1,3),(3,2),(2,3)]
    E2 = [(3,4),(1,2),(1,3),(4,1),(4,2),(2,4),(3,2),(2,3),(2,1),(1,4),(4,3),(3,1)]
    E3 = [(1,4),(4,2),(4,5),(6,3),(3,5),(4,6),(5,6),(3,2),(3,6),(6,5),(5,2),(2,4),
          (3,1),(5,3),(4,3),(2,6),(5,4),(4,1),(5,1),(2,3)]
    for i in range(len(E1)): G1.addEdge(E1[i][0],E1[i][1],len(E1)-i)
    for i in range(len(E2)): G2.addEdge(E2[i][0],E2[i][1],len(E2)-i)
    for i in range(len(E3)): G3.addEdge(E3[i][0],E3[i][1],len(E3)-i)
    S1 = "1 => (3,3)\n3 => (1,3)"
    S2 = "1 => (2,4) (4,3)\n2 => (1,4) (4,7) (3,5)\n3 => (4,2) (2,5)\n4 => (3,2) (1,3) (2,7)"
    G1.deleteNode(2)
    G2.deleteEdge(3,1)
    print(chr(10209)+" Test Case 1:","PASSED" if str(G1) == S1 else "FAILED")
    print(chr(10209)+" Test Case 3:","PASSED" if G2.hasEdge(2,3) else "FAILED")
    print(chr(10209)+" Test Case 4:","PASSED" if not G3.hasEdge(2,1) else "FAILED")
    print(chr(10209)+" Test Case 5:","PASSED" if str(G2) == S2 else "FAILED")
    print(chr(10209)+" Test Case 6:","PASSED" if G3.getWeight(2,4) == 9 else "FAILED")

if __name__ == '__main__': TESTGRAPH()
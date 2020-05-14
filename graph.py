'''
GRAPH Class

Represents an abstract weighted & undirected graph using a set of nodes and
a dictionary of edges, where the keys are the tuple of connected nodes and the
values are the weights of the connection. Use .addNode() to add a new node
and .addEdge() to connect 2 nodes. Use .neighborDict() to get a dictionary of
all nodes connected to each node.
'''

class GRAPH(object):
    def __init__(self):
        self.nodes = set()
        self.edges = dict()

    # Adds node to the graph
    def addNode(self,node):
        self.nodes.add(node)

    # Adds an edge between node1 and node2 with a weight of weight
    def addEdge(self,node1,node2,weight):
        self.edges[(node1,node2)] = weight
        self.edges[(node2,node1)] = weight

    # Returns a dictionary of all nodes and the set of their neighbors
    def neighborDict(self):
        nDict = {}
        for node in self.nodes:
            nDict[node] = set()
        for edge in self.edges:
            node1, node2 = edge
            nDict[node1].add(node2)
            nDict[node2].add(node1)
        return nDict
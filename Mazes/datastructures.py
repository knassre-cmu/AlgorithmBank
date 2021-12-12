import functools, random, copy, math, numpy

class RedBlack(object):
    __doc__ = '''
RedBlack:

A binary search tree class using the Red Black scheme to ensure that the tree
is aproximately balanced so that the height is logarithmic. Used to implement
a dictionary interface.
'''

    # Node class used to contain each branch/leaf of the BST
    @functools.total_ordering
    class Node(object):
        # Each node has a key-value pair, a color (red or black), and stores
        # the nodes on the left and right (which could be None)
        __match_args__ = ("key", "value", "color", "left", "right")
        def __init__(self, key, value, color, left, right):
            self.key = key
            self.value = value
            self.color = color
            self.left = left
            self.right = right

        def __repr__(self):
            L = "L" if self.left == None else self.left
            R = "L" if self.right == None else self.right
            return f"{self.color}({L}, {self.key}->{self.value}, {R})"
        
        # Nodes can be compared, thus allowing them to be sorted
        def __lt__(self, other):
            return repr(self.key) < repr(other.key)
        
        # Inserts a node into a node, either replacing its value or injecting
        # it on its left/right, thus maintaining BST ordering and RBT invariants
        def insert(self, newNode):
            if self.key == newNode.key:
                self.value = newNode.value
            elif newNode < self:
                if self.left == None:
                    self.left = newNode
                else:
                    self.left.insert(newNode)
                    self.balance()
            else:
                if self.right == None:
                    self.right = newNode
                else:
                    self.right.insert(newNode)
                    self.balance()

        # Uses pattern matching to identify cases where rotations need to take
        # place in order to maintain RBT invariants
        def balance(self):
            match self:
                case (RedBlack.Node(kZ, vZ, "B", RedBlack.Node(kY, vY, "R", RedBlack.Node(kX, vX, "R", a, b), c), d)
                |  RedBlack.Node(kZ, vZ, "B", RedBlack.Node(kX, vX, "R", a, RedBlack.Node(kY, vY, "R", b, c)), d)
                |  RedBlack.Node(kX, vX, "B", a, RedBlack.Node(kZ, vZ, "R", RedBlack.Node(kY, vY, "R", b, c), d))
                |  RedBlack.Node(kX, vX, "B", a, RedBlack.Node(kY, vY, "R", b, RedBlack.Node(kZ, vZ, "R", c, d)))):
                    self.key, self.value = kY, vY
                    self.color = "R"
                    self.left = RedBlack.Node(kX, vX, "B", a, b)
                    self.right = RedBlack.Node(kZ, vZ, "B", c, d)
        
        # Allows the entire tree to be looped over in in-order traversal by 
        # recursively looping over the children
        def __iter__(self):
            if self.left != None:
                yield from self.left
            yield self.key
            if self.right != None:
                yield from self.right

    # Initializes the RBT dictionary, which can take in initial values via kwargs
    def __init__(self, **kwargs):
        self.root = None
        self.size = 0
        for key in kwargs:
            self[key] = kwargs[key]

    def __repr__(self):
        return repr(self.root)

    # Obtains the vaue for a certain key, or None if it is not present
    def __getitem__(self, key):
        node = self.root
        while node != None:
            if key == node.key: return node.value
            elif key < node.key: node = node.left
            else: node = node.right
        return None

    # Returns True if a certain key is in the RBT
    def __contains__(self, key):
        node = self.root
        while node != None:
            if key == node.key: return True
            elif key < node.key: node = node.left
            else: node = node.right
        return False

    # Adds or replaces a key-value pair in the RBT
    def __setitem__(self, key, value):
        if self.root == None:
            self.root =  RedBlack.Node(key, value, "B", None, None)
            self.size = 1
        else:
            if key not in self: self.size += 1
            self.root.insert(RedBlack.Node(key, value, "R", None, None))
            self.root.color = "B"

    # Loops over the keys in the RBT
    def __iter__(self):
        if self.root != None:
            yield from self.root

    # Returns the number of keys in the RBT
    def __len__(self):
        return self.size

    # Returns True if two RBTs are equivilent
    def __eq__(self, other):
        return isinstance(other, RedBlack) and self.items() == other.items()

    # Allows RBTs to interact with **kwargs
    def keys(self):
        return iter(self)

    # Returns all values in the RBT
    def values(self):
        values = []
        stack = [self.root]
        while stack:
            node = stack.pop()
            if node == None: continue
            elif not isinstance(node, RedBlack.Node):
                values.append(node)
            else:
                stack.append(node.right)
                stack.append(node.value)
                stack.append(node.left)
        return values

    # Returns all key-value pairs in the RBT
    def items(self):
        items = []
        stack = [self.root]
        while stack:
            node = stack.pop()
            if node == None: continue
            elif not isinstance(node, RedBlack.Node):
                items.append(node)
            else:
                stack.append(node.right)
                stack.append((node.key, node.value))
                stack.append(node.left)
        return items

    # Nondestructively maps a function over all key-value pairs in an RBT
    def map(self, fn):
        result = RedBlack()
        for key, val in self.items():
            newKey, newVal = fn(key, val)
            result[newKey] = newVal
        return result

    # Nondestructively filters the key-value pairs in an RBT by a function
    def filter(self, fn):
        result = RedBlack()
        for key, val in self.items():
            if fn(key, val):
                result[key] = val
        return result

    # Nondestructively combines all values in an RBT with a function
    def reduce(self, fn, seed):
        result = seed
        for key, val in self.items():
            result = fn(result, key, val)
        return result

    # Nondestructively merges two RBTs
    def __add__(self, other):
        result = RedBlack()
        for key, val in self.items():
            result[key] = val
        for key, val in other.items():
            result[key] = val
        return result

    # Nondestructively removes all keys from one RBT that are in another
    def __sub__(self, other):
        result = RedBlack()
        for key, val in self.items():
            if key not in other:
                result[key] = val
        return result

def testRedBlack():
    print("Testing RedBlack...", end="")
    
    # Creating with kwargs
    R = RedBlack(A=1, B=5, C=1, D=1, E=2, F=15, G=1, H=12, J=100)
    S = RedBlack(A=65, E=69, I=73, O=76, U=85, Y=89)
    
    # Getting and setting items (plus length checking)
    assert(len(R) == 9)
    assert(R["A"] == 1)
    assert(R["I"] == None)
    R["I"] = 15
    R["J"] = 112
    assert(R["I"] == 15)
    assert(R["J"] == 112)
    assert(len(R) == 10)
    
    # Converting to iterables/mappings
    assert(list(R) == list("ABCDEFGHIJ"))
    assert(dict(R) == {'A': 1, 'B': 5, 'C': 1, 'D': 1, 'E': 2, 'F': 15, 'G': 1, 'H': 12, 'I': 15, 'J': 112})
    
    # Equality checking
    assert(R != S)
    assert(R != RedBlack(A=1, B=5, C=1, D=1, E=2, F=15, G=1, H=12, I=15, J=100))
    assert(R == RedBlack(A=1, B=5, C=1, D=1, E=2, F=15, G=1, H=12, I=15, J=112))
    assert(R != dict(R))
    assert(R != "Fudge")

    # Extarcting keys, values, and items
    assert(R.values() == [1, 5, 1, 1, 2, 15, 1, 12, 15, 112])
    assert(R.items() == [('A', 1), ('B', 5), ('C', 1), ('D', 1), ('E', 2), ('F', 15), ('G', 1), ('H', 12), ('I', 15), ('J', 112)])
    
    # Using HOFs
    assert(R.map(lambda k, v: (2*k, v**2)) == RedBlack(AA=1, BB=25, CC=1, DD=1, EE=4, FF=225, GG=1, HH=144, II=225, JJ=12544))
    assert(R.filter(lambda k, v: ord(k) % 2 == 1) == RedBlack(A=1, C=1, E=2, G=1, I=15))
    assert(R.reduce(lambda acc, k, v: int(str(acc) + str(v)), 0) == 151121511215112)
    
    # Merging with addition
    assert(R + S == RedBlack(A=65, B=5, C=1, D=1, E=69, F=15, G=1, H=12, I=73, J=112, O=76, U=85, Y=89))
    assert(S + R == RedBlack(A=1, B=5, C=1, D=1, E=2, F=15, G=1, H=12, I=15, J=112, O=76, U=85, Y=89))
    
    # Removing with subtraction
    assert(R - S == RedBlack(B=5, C=1, D=1, F=15, G=1, H=12, J=112))
    assert(S - R == RedBlack(O=76, U=85, Y=89))
    print("Passed!")

class PriorityQueue(object):
    __doc__ = '''
Priority Queue:

A priority queue class using an array to implement a minheap. Allows for
logarithmic insertions and removals of the minimum element.
'''

    # Initializes the PQ, which can have initial values taken in via *args.
    # The comparison funciton can also be provided via **kwargs
    def __init__(self, *args, **kwargs):
        self.cmp = kwargs.get("cmp", lambda x, y: x < y)
        self.arr = [None]
        for arg in args:
            self.push(arg)
    
    # Returns the number of elements in the PQ
    def __len__(self):
        return len(self.arr) - 1

    # Yields each element in the PQ in sorted order
    def __iter__(self):
        L = []
        for i in range(len(self)):
            L.append(self.pop())
        for elem in L:
            self.push(elem)
        yield from L


    def __repr__(self):
        return f"PriorityQueue({', '.join([repr(elem) for elem in self])})"

    # Returns True if 2 PQ are equivilent
    def __eq__(self, other):
        if not isinstance(other, PriorityQueue):
            return False
        d1 = {}
        d2 = {}
        for elem in self:
            d1[elem] = d1.get(elem, 0) + 1
        for elem in other:
            d2[elem] = d2.get(elem, 0) + 1
        return d1 == d2
    
    # Nondestructively combines 2 PQ, using the former's comparison function
    def __add__(self, other):
        return PriorityQueue(*self, *other, cmp=self.cmp)

    # Adds an element to the PQ
    def push(self, elem):
        self.arr.append(elem)
        i = len(self.arr)-1
        while i > 1:
            if self.cmp(elem, self.arr[i//2]):
                self.arr[i], self.arr[i//2] = self.arr[i//2], self.arr[i]
                i //= 2
            else: break

    # Removes the minimum element from the PQ
    def pop(self):
        if len(self.arr) < 2:
            raise Exception("Cannot pop from empty PQ")
        if len(self.arr) == 2: 
            return self.arr.pop()
        elem = self.arr[1]
        self.arr[1] = self.arr.pop()
        i = 1
        while 2*i < len(self.arr):
            if 2*i + 1 >= len(self.arr) or self.cmp(self.arr[2*i], self.arr[2*i+1]):
                if self.cmp(self.arr[2*i], self.arr[i]):
                    self.arr[i], self.arr[2*i] = self.arr[2*i], self.arr[i]
                    i *= 2
                else: break
            else:
                if self.cmp(self.arr[2*i+1], self.arr[i]):
                    self.arr[i], self.arr[2*i+1] = self.arr[2*i+1], self.arr[i]
                    i *= 2
                else: break
        return elem

    # Nondestructively returns the kth largest element in the PQ
    def __getitem__(self, k):
        if k < 0 or k >= len(self): 
            return None
        L = []
        for i in range(k+1):
            L.append(self.pop())
        for elem in L:
            self.push(elem)
        return L[-1]

def testPriorityQueue():
    print("Testing PriorityQueue...", end="")
    P = PriorityQueue(1, 5, 1, 1, 2)
    assert(len(P) == 5)
    assert(list(P) == [1, 1, 1, 2, 5])
    assert(P[0] == 1)
    assert(P[1] == 1)
    assert(P[2] == 1)
    assert(P[3] == 2)
    assert(P[4] == 5)
    P.push(42)
    P.push(0)
    assert(list(P) == [0, 1, 1, 1, 2, 5, 42])
    assert([P.pop() for i in range(3)] == [0, 1, 1])
    assert(list(P) == [1, 2, 5, 42])
    Q = PriorityQueue(15, 112, 42, cmp=lambda x, y: str(x) < str(y))
    assert(list(Q) == [112, 15, 42])
    R = P + Q
    assert(list(R) == [1, 2, 5, 15, 42, 42, 112])
    S = Q + P
    assert(list(S) == [1, 112, 15, 2, 42, 42, 5])
    assert(P != Q)
    assert(R == S)
    assert(R != 42)
    print("Passed!")

class Graph(object):
    __doc__ = '''
Graph:

A graph class that uses an adjacency dictionary implementation. By default
implements weighted undirected graphs, but can be modified to implement
directed graphs, and the weights can be ignored in order to achieve an
unweighted graph. Accompanied by a wide array of graph aglrithms.
'''

    # Initializes the graph, which is undirected by default
    def __init__(self, **kwargs):
        self.table = {}
        self.directed = kwargs.get("directed", False)

    # Adds a node to the graph, if not already included
    def addNode(self, v):
        self.table[v] = self.table.get(v, {})

    # Adds an edge to the graph, and the reverse edge if unweighted
    def addEdge(self, u, v, w):
        self.addNode(u)
        self.addNode(v)
        self.table[u][v] = w
        if not self.directed: self.table[v][u] = w

    # Removes an edge from the grpah, and the reverse edge if unweighted
    def removeEdge(self, u, v):
        if v in self.table.get(u, {}): self.table[u].pop(v)
        if not self.directed and u in self.table.get(v, {}): self.table[v].pop(u)

    # Returns the number of nodes in a graph
    def __len__(self):
        return len(self.table)

    # Returns the weight of an edge in the graph, or 0 if there is no edge
    def __getitem__(self, pair):
        u, v = pair
        return self.table.get(u, {}).get(v, 0)

    # Creates/updates the weight of an edge in the graph
    def __setitem__(self, pair, w):
        u, v = pair
        self.addEdge(u, v, w)

    # Loops over all nodes in a graph
    def __iter__(self):
        yield from self.table

    # Returns a set of all nodes in a graph
    def getNodes(self):
        return set(self.table)

    # Returns a set of all neighbors of a node in a graph
    def getNeighbors(self, v):
        return set(self.table.get(v, {}))

    # Returns a set of all edges in a graph
    def getEdges(self):
        edges = set()
        for u in self.table:
            for v in self.table[u]:
                w = self.table[u][v]
                if not self.directed and (w, v, u) in edges: continue
                edges.add((w, u, v))
        return edges

    # Returns the set of all nodes connected to a certain node using DFS
    def getConenctedNodes(self, node):
        component = {node}
        visited = {node}
        def search(node):
            for neighbor in self.table[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    component.add(neighbor)
                    search(neighbor)
        search(node)
        return component

    # Returns a list of connected components in the graph using DFS
    def getConnectedComponents(self):
        components = []
        visited = set()
        def search(node, component):
            for neighbor in self.table[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    component.append(neighbor)
                    search(neighbor, component)
        for node in self.table:
            if node in visited: continue
            visited.add(node)
            component = [node]
            search(node, component)
            components.append(component)
        return components

    # Uses Breadth First Search to find a path between 2 nodes in a graph, or
    # None if no path exists
    def breadthFirstSearch(self, start, target):
        prev = {start: None}
        queue = [start]
        visited = {start}
        unvisitedNeighbors = set()
        while len(queue) > 0:
            node = queue.pop(0)
            if node == target: break
            visited.add(node)
            for neighbor in self.table[node]:
                if neighbor in visited or neighbor in unvisitedNeighbors: continue
                prev[neighbor] = node
                queue.append(neighbor)
                unvisitedNeighbors.add(neighbor)
        if target not in prev: return None
        path = []
        node = target
        while node != None:
            path.append(node)
            node = prev[node]
        return path [::-1]

    # Uses Dijkstra's algorithm to find the shortest path in a graph, or
    # None if no such path exists
    def dijkstra(self, start, target):
        return self.aStar(start, target, lambda x: 0)

    # Uses A* with the heuristic provided to find the shortest path in a graph,
    # or None if no such path exists
    def aStar(self, start, target, heuristic):
        prev = {start: None}
        dist = {node: float("inf") for node in self.table}
        pq = PriorityQueue((0, start))
        visited = set()
        dist[start] = 0
        while len(pq) > 0:
            _, node = pq.pop()
            if node == target: break
            if node in visited: continue
            visited.add(node)
            for neighbor in self.table[node]:
                oldDist = dist[neighbor]
                newDist = dist[node] + self.table[node][neighbor]
                if newDist < oldDist:
                    prev[neighbor] = node
                    dist[neighbor] = newDist
                    pq.push((newDist + heuristic(neighbor), neighbor))
        if target not in prev: return None
        path = []
        node = target
        while node != None:
            path.append(node)
            node = prev[node]
        return path [::-1]

    # Uses Kruskal's algorithm to find a minimum spanning tree of a graph
    def kruskal(self):
        edges = sorted(self.getEdges())
        mst = Graph()
        edgesAdded = 0
        ufs = {node: node for node in self.table}
        def find(node):
            if ufs[node] == node: return node
            rep = find(ufs[node])
            ufs[node] = rep
            return rep
        def union(a, b):
            if random.randint(0, 1):
                ufs[a] = b
            else:
                ufs[b] = a
        for w, u, v in edges:
            a, b = find(u), find(v)
            if a == b: continue
            mst.addEdge(u, v, w)
            edgesAdded += 1
            if edgesAdded == len(self.table)-1: break
            union(a, b)
        return mst

    # Uses the Edmond-Karp algorithm to find the maximum flow in a graph
    # between two nodes, as well as return the flow graph
    def edmondKarp2(self, source, sink):
        G = copy.deepcopy(self)
        F = Graph(directed=True)
        for node in G: 
            F.addNode(node)
        maxFlow = 0
        while True:
            path = G.breadthFirstSearch(source, sink)
            if path == None: break
            flow = float("inf")
            for i in range(len(path)-1):
                u, v = path[i], path[i+1]
                w = G[u, v]
                flow = min(flow, w)
            maxFlow += flow
            for i in range(len(path)-1):
                u, v = path[i], path[i+1]
                F[u, v] += flow
                G[u, v] -= flow
                G[v, u] += flow
                if G[u, v] == 0:
                    G.removeEdge(u, v)
        nodes = F.getNodes()
        for u in nodes:
            for v in nodes:
                w, r = F[u, v], F[v, u]
                if w != 0 and r != 0:
                    if w > r:
                        F[u, v] -= r
                        F.removeEdge(v, u)
                    elif w < r:
                        F[v, u] -= w
                        F.removeEdge(u, v)
                    else:
                        F.removeEdge(u, v)
                        F.removeEdge(v, u)
        return maxFlow, F
            
def testGraph():
    print("Testing Graph...", end="")

    # Creating an empty undirected Graph
    A = Graph()
    assert(len(A) == 0)
    assert(A.getNodes() == set())
    assert(A.getNeighbors("x") == set())
    assert(A.getNeighbors("y") == set())
    assert(A.getEdges() == set())
    assert(A["x", "y"] == 0)
    assert(A["y", "z"] == 0)
    assert(A.getConnectedComponents() == [])

    # Adding nodes and edges to a Graph (and connected components)
    A["x", "y"] = 1
    A["x", "z"] = 2
    A["a", "b"] = 3
    A.addNode("c")
    assert(len(A) == 6)
    assert(A.getNodes() == {"a", "b", "c", "x", "y", "z"})
    assert(A.getNeighbors("x") == {"y", "z"})
    assert(A.getNeighbors("y") == {"x"})
    assert(A.getEdges() == {(1, "x", "y"), (2, "x", "z"), (3, "a", "b")})
    assert(A.getConenctedNodes("x") == {"x", "y", "z"})
    assert(sorted([sorted(comp) for comp in A.getConnectedComponents()])
           == [['a', 'b'], ['c'], ['x', 'y', 'z']])

    # Updating and removing edges (and connected components)
    A["x", "z"] += 13
    A.removeEdge("x", "y")
    assert(A.getNeighbors("x") == {"z"})
    assert(A.getNeighbors("y") == set())
    assert(A.getEdges() == {(15, "x", "z"), (3, "a", "b")})
    assert(A.getConenctedNodes("x") == {"x", "z"})
    assert(sorted([sorted(comp) for comp in A.getConnectedComponents()])
           == [['a', 'b'], ['c'], ['x', 'z'], ['y']])

    # Testing pathfinding and minimum spanning trees on undirected graphs
    B = Graph()
    edges = [("a", "b", 1.0),
             ("a", "c", 2.0),
             ("a", "d", 3.0),
             ("a", "e", 4.0),
             ("b", "c", 1.3),
             ("c", "d", 1.6),
             ("d", "e", 1.5),
             ("c", "f", 3.9),
             ("d", "f", 1.1),
             ("g", "h", 0.8),
             ("g", "i", 2.2),
             ("h", "i", 3.5)]
    for u, v, w in edges:
        B[u, v] = w

    assert(B.breadthFirstSearch("h", "i") == ["h", "i"])
    assert(B.breadthFirstSearch("a", "b") == ["a", "b"])
    assert(B.breadthFirstSearch("b", "f") == ["b", "c", "f"])

    assert(B.dijkstra("h", "i") == ["h", "g", "i"])
    assert(B.dijkstra("a", "b") == ["a", "b"])
    assert(B.dijkstra("b", "f") == ["b", "c", "d", "f"])

    ordHeuristic = lambda c: lambda x: abs(ord(c) - ord(x))
    assert(B.aStar("h", "i", ordHeuristic("i")) == ["h", "g", "i"])
    assert(B.aStar("a", "b", ordHeuristic("i")) == ["a", "b"])
    assert(B.aStar("b", "f", ordHeuristic("i")) == ["b", "c", "d", "f"])

    B["g", "a"] = 5.5
    M = B.kruskal()
    assert(M.getEdges() == {(1.6, 'd', 'c'), (1.3, 'b', 'c'), (5.5, 'g', 'a'), 
                            (0.8, 'g', 'h'), (1.0, 'a', 'b'), (2.2, 'g', 'i'), 
                            (1.5, 'd', 'e'), (1.1, 'd', 'f')})

    # Testing directed graphs
    C = Graph(directed=True)
    edges = [("a", "b", 1),
             ("a", "c", 2),
             ("a", "d", 3),
             ("b", "e", 4),
             ("e", "d", 5),
             ("d", "c", 6),
             ("d", "b", 7),
             ("b", "a", 8)]
    for u, v, w in edges:
        C[u, v] = w
    assert(C.getNodes() == {"a", "b", "c", "d", "e"})
    assert(C.getNeighbors("d") == {"b", "c"})
    assert(C.getNeighbors("c") == set())
    assert(C.getEdges() == {(1, "a", "b"), (2, "a", "c"), (3, "a", "d"),
                            (4, "b", "e"), (5, "e", "d"), (6, "d", "c"),
                            (7, "d", "b"), (8, "b", "a")})
    assert(C["a", "b"] == 1)
    assert(C["b", "a"] == 8)
    assert(C.getNeighbors("a") == {"b", "c", "d"})
    assert(C.getNeighbors("b") == {"a", "e"})

    C.removeEdge("a", "b")
    assert(C["a", "b"] == 0)
    assert(C["b", "a"] == 8)
    assert(C.getNeighbors("a") == {"c", "d"})
    assert(C.getNeighbors("b") == {"a", "e"})

    # Testing Max Flow
    D = Graph(directed=True)
    edges = [("s", "a", 4),
             ("s", "b", 2),
             ("a", "e", 3),
             ("b", "c", 2),
             ("b", "d", 3),
             ("c", "b", 1),
             ("c", "t", 2),
             ("d", "g", 4),
             ("e", "f", 3),
             ("f", "c", 3),
             ("g", "t", 4)]
    for u, v, w in edges:
        D[u, v] = w

    flow, flowGraph = D.edmondKarp2("s", "t")
    assert(flow == 5)
    assert(flowGraph.getEdges() == {(3, 'd', 'g'), (3, 's', 'a'), (1, 'c', 'b'), 
                                    (3, 'g', 't'), (3, 'a', 'e'), (2, 's', 'b'), 
                                    (3, 'b', 'd'), (3, 'f', 'c'), (2, 'c', 't'), 
                                    (3, 'e', 'f')})

    print("Passed!")

class Polygon(object):
    __doc__ = '''
Polygon:

A polygon class which can store a list of (x, y) points representing a polygonal
object in the xy plane, as well as several computational geometry algorithms.
'''

    # Initializes the polygon using points taken in from *args
    def __init__(self, *points):
        self.points = [[x, y] for x, y in points]

    def __repr__(self):
        return f"Polygon({', '.join([repr(point) for point in self.points])})"

    # Returns the number of points in the polygon
    def __len__(self):
        return len(self.points)

    # Loops over the points in a polygon
    def __iter__(self):
        for x, y in self.points:
            yield x, y

    # Performs line-side tests to see if a point is inside of a polygon
    # (assumes points are in counterclockwise order)
    def __contains__(self, point):
        x, y = point
        for i in range(len(self.points)):
            x1, y1 = self.points[i]
            x2, y2 = self.points[(i+1)%len(self.points)]
            if self.lineSideTest(x1, y1, x2, y2, x, y) == "R":
                return False
        return True

    # Shifts the polygon by a certain x and y amount
    def translate(self, dx, dy):
        for point in self.points:
            point[0] += dx
            point[1] += dy

    # Returns L, O, or R depending on if (px, py) is on the left of the line
    # segment (x1, y1) - (x2, y2), or on the segment, or to its right
    @staticmethod
    def lineSideTest(x1, y1, x2, y2, px, py):
        ax = x2-x1
        ay = y2-y1
        bx = px-x1
        by = py-y1
        cross = ax*by - ay*bx
        if cross > 0: return "L"
        elif cross < 0: return "R"
        else: return "O"

    # Returns the standard form (A, B, C) of a line segment formed by 2 points
    @staticmethod
    def standardForm(x0, y0, x1, y1):
        if x0 == x1:
            return 1, 0, x0
        elif y0 == y1:
            return 0, 1, y0
        else:
            m = (y1 - y0) / (x1 - x0)
            return -m, 1, y0 - x0*m

    # Uses the Graham Scan to create a convex hull
    @staticmethod
    def convexHull(*points):
        origin = min(sorted(points), key=lambda p: p[1])
        def t(x, y):
            if (x, y) == origin: return 0
            else: return math.atan2(y-origin[1], x-origin[0])
        def r(x, y):
            return ((x-origin[0])**2 + (y-origin[1])**2)**0.5
        points = [(t(x, y), r(x, y), x, y) for x, y in points]
        points.sort(key=lambda p: p[0])
        stack = points[:2]
        for i in range(2, len(points)):
            while True:
                _, __, ax, ay = stack[-2]
                _, br, bx, by = stack[-1]
                _, cr, cx, cy = points[i]
                match (Polygon.lineSideTest(ax, ay, bx, by, cx, cy)):
                    case "L":
                        stack.append(points[i])
                        break
                    case "O":
                        if br <= cr:
                            stack[-1] = points[i]
                        break
                    case "R":
                        stack.pop()
        return Polygon(*[(x, y) for _, _, x, y in stack])

    # Returns true if 2 line segments are intersecting
    @staticmethod
    def lineIntersection(x0, y0, x1, y1, x2, y2, x3, y3):
        a = Polygon.lineSideTest(x0, y0, x1, y1, x2, y2)
        b = Polygon.lineSideTest(x0, y0, x1, y1, x3, y3)
        c = Polygon.lineSideTest(x2, y2, x3, y3, x0, y0)
        d = Polygon.lineSideTest(x2, y2, x3, y3, x1, y1)
        match (a, b, c, d):
            case ("L", "L", _, _) | ("R", "R", _, _) | (_, _, "L", "L") | (_, _, "R", "R"):
                return False
            case _:
                return True

    # Returns True if 2 polygons are overlapping by checking if any point
    # from either polygon is inside of the other or if any of their line
    # segments are intersecting
    def overlapping(self, other):
        for point in self:
            if point in other: return True
        for point in other:
            if point in self: return True
        for i in range(len(self.points)):
            x0, y0 = self.points[i]
            x1, y1 = self.points[(i+1)%(len(self.points))]
            for j in range(len(other.points)):
                x2, y2 = self.points[j]
                x3, y3 = self.points[(j+1)%(len(self.points))]
                if self.lineIntersection(x0, y0, x1, y1, x2, y2, x3, y3): 
                    return True
        return False

def testPolygon():
    print("Testing Polygon...", end="")
    # Testing basic properties of Polygons
    P = Polygon((0, 0), (4, 0), (2, 4))
    assert(len(P) == 3)
    assert(list(P) == [(0, 0), (4, 0), (2, 4)])
    P.translate(1, -1)
    assert(list(P) == [(1, -1), (5, -1), (3, 3)])

    # Testing line-side test
    assert(P.lineSideTest(5, 0, 5, 5, 0, 3) == "L")
    assert(P.lineSideTest(5, 0, 5, 5, 5, 3) == "O")
    assert(P.lineSideTest(5, 0, 5, 5, 7, 3) == "R")
    assert(P.lineSideTest(5, 0, 5, 5, 0, 13) == "L")
    assert(P.lineSideTest(5, 0, 5, 5, 5, 13) == "O")
    assert(P.lineSideTest(5, 0, 5, 5, 7, 13) == "R")
    assert(P.lineSideTest(5, 0, 5, 5, 0, -20) == "L")
    assert(P.lineSideTest(5, 0, 5, 5, 5, -20) == "O")
    assert(P.lineSideTest(5, 0, 5, 5, 7, -20) == "R")
    assert(P.lineSideTest(0, 10, 10, 0, 6, 6) == "L")
    assert(P.lineSideTest(0, 10, 10, 0, 5, 5) == "O")
    assert(P.lineSideTest(0, 10, 10, 0, 4, 4) == "R")

    # Testing standard form
    assert(P.standardForm(0, -4, 2, 0) == (-2, 1, -4))
    assert(P.standardForm(8, 7, 8, 3) == (1, 0, 8))
    assert(P.standardForm(5, 2, 9, 2) == (0, 1, 2))

    # Testing point containment
    assert((3, -2) not in P)
    assert((3, -1) in P)
    assert((3, 0) in P)
    assert((3, 1) in P)
    assert((3, 2) in P)
    assert((3, 3) in P)
    assert((3, 4) not in P)
    assert((3, 0) in P)
    assert((2, 0) in P)
    assert((1, 0) not in P)
    assert((4, 0) in P)
    assert((5, 0) not in P)

    # Testing convex hull maker
    Q = Polygon.convexHull((-5, -8), (0, -5), (-2, 1), (-1, 0), (5, -2), (-3, -6), 
                           (3, 3), (2, -2), (4, 3), (1, -1), (-4, -2))
    assert(list(Q) == [(-5, -8), (5, -2), (4, 3), (3, 3), (-2, 1), (-4, -2)])

    # Testing line intersection
    assert(Polygon.lineIntersection(0, 2, 4, 0, 0, 0, 4, 4) == True)
    assert(Polygon.lineIntersection(4, 0, 0, 2, 0, 0, 4, 0) == True)
    assert(Polygon.lineIntersection(0, 0, 4, 0, 3, 1, 2, 0) == True)
    assert(Polygon.lineIntersection(4, 0, 4, 4, 0, 2, 0, 0) == False)
    assert(Polygon.lineIntersection(0, 0, 0, 2, 4, 0, 4, 4) == False)
    assert(Polygon.lineIntersection(0, 0, 0, 2, -4, 3, 4, 3) == False)
    assert(Polygon.lineIntersection(0, 0, 0, 2, -4, 1, 4, 1) == True)
    print("Passed!")

if __name__ == "__main__":
    testRedBlack()
    testPriorityQueue()
    testGraph()
    testPolygon()

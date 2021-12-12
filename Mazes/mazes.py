from Graphics import *
from noise import Perlin
from datastructures import Graph
import random, math, copy

class Mazes(App):
    def appStarted(self):
        self.main = MainMode()
        self.setMode("main")

class MainMode(Mode):
    def appStarted(self):
        self.gridSize = min(self.width, self.height) * 0.5
        self.algorithms = {
            "Recursive Backtracking": (self.dfs, 21, 21), # Must be odd dims
            "Prim's Algorithm": (self.prim, 21, 21), # Must be odd dims
            "Kruskal's Algorithm": (self.kruskal, 21, 21), # Must be odd dims
            "Eller's Algorithm": (self.eller, 21, 21), # Must be odd dims
            "Hunt & Kill": (self.hunt, 21, 21),  # Must be odd dims
            "Islamic City": (self.islamic, 24, 24), # Can have any dims
            "Kruskal Miner": (self.miner, 150, 150), # Can have any dims, ideally large
            "Sine Waves": (self.sine, 100, 100), # Can have any dims
            "Cellular Automata": (self.automata, 75, 75), # Ideally large dims
            "Voronoi Noise": (self.voronoi, 64, 64), # Can have any dims
            "Diamond Square": (self.diamondSquare, 129, 129), # Square power of 2 plus 1
            "Perlin Noise": (self.perlin, 100, 100), # Procedural generation :)
            "Pacman Grid": (self.pacrat, 31, 31), # Must have multiple of 4 - 1 dims
            "Twisted Pipes": (self.pipes, 7, 7), # Can have any dims
            "Killer Sudoku": (self.killer, 9, 9), # Square, ideally small-ish
            "Sliding Tile": (self.sliding, 5, 5), # Ideally small-ish
            "Kami Puzzle": (self.kami, 50, 50), # Procedural generation :)
            }
        self.algorithmQueue = list(self.algorithms)
        self.currentAlgorithm = self.algorithmQueue[0]
        self.makeMaze()
        self.pipeAlgos = {"Twisted Pipes"}
        self.colorAlgos = {"Killer Sudoku", "Sliding Tile", "Kami Puzzle"}
        self.isometric = False

    def shiftAlgorithm(self, reverse=False):
        if reverse:
            self.algorithmQueue.insert(0, self.algorithmQueue.pop())
        else:
            self.algorithmQueue.append(self.algorithmQueue.pop(0))
        self.currentAlgorithm = self.algorithmQueue[0]
        self.makeMaze()

    def makeMaze(self):
        algorithm, self.rows, self.cols = self.algorithms[self.currentAlgorithm]
        self.maze = algorithm(self.rows, self.cols)

    # Helper utility for other algorithms: obtain set of all nodes connected
    # to a seed node within a grid using a stack-based DFS
    def getConnected(self, rows, cols, grid, seed):
        reachable = set()
        stack = [seed]
        while len(stack) > 0:
            row, col = stack.pop()
            reachable.add((row, col))
            for nr, nc in [(row, col-1), (row, col+1), (row-1, col), (row+1, col)]:
                if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 1 and (nr, nc) not in reachable:
                    stack.append((nr, nc))
        return reachable

    # Helper utility for other algorithms: smooths out a noise grid slightly by
    # adjusting a cell based on its neighbors
    def smoothGrid(self, rows, cols, grid):
        newGrid = copy.deepcopy(grid)
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == 0: continue
                cells = [grid[r][c]]
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r+dr, c+dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        cells.append(grid[nr][nc])
                newGrid[r][c] = 0 if cells.count(0) > len(cells)//2 else 1
        return newGrid

    # Recursive DFS to generate a maze
    def dfs(self, rows, cols):
        
        # Create the grid and the (initially empty) set of visited nodes
        grid = [[0 for c in range(cols)] for r in range(rows)]
        visited = set()

        # Recursively generate the maze from a certain current (row, col) by
        # looping over its neighbors in a random order, adding a connection
        # if they are unvisited, and recursively generating from there
        def recursiveBacktracker(r, c):
            visited.add((r, c))
            neighbors = [(r-2, c), (r+2, c), (r, c-2), (r, c+2)]
            random.shuffle(neighbors)
            for nr, nc in neighbors:
                if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited:
                    grid[r][c] = 1
                    grid[(r+nr)//2][(c+nc)//2] = 1
                    grid[nr][nc] = 1
                    recursiveBacktracker(nr, nc)

        # Invoke the recursive procedure from a start node and return the grid
        recursiveBacktracker(0, 0)
        return grid

    # Prim's algorithm used for maze generation
    def prim(self, rows, cols):
        # Create the grid, start node, set of nodes in the maze, list of nodes
        # that neighbor a node in the maze, and number of edges added so far
        grid = [[0 for c in range(cols)] for r in range(rows)]
        seed = (0, 0)
        visited = {seed}
        unvisitedNeighbors = [seed]
        i = 0

        # Loop until the number of edges added so far is such that every node
        # must be connected
        while i < ((rows//2+1) * (rows//2+1)) - 1:

            # Extract a random node from the unvisited neighbors
            index = random.randint(0, len(unvisitedNeighbors)-1)
            r, c = unvisitedNeighbors.pop(index)

            # Loop over its unvisited neighbors
            neighbors = [(r-2, c), (r+2, c), (r, c-2), (r, c+2)]
            for nr, nc in neighbors:
                if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited:

                    # Add the edge and add the neighbors to the list
                    grid[r][c] = 1
                    grid[(r+nr)//2][(c+nc)//2] = 1
                    grid[nr][nc] = 1
                    i += 1
                    visited.add((nr, nc))
                    unvisitedNeighbors.append((nr, nc))

        return grid

    # Kruskal's algorithm used for maze generation
    def kruskal(self, rows, cols):
        # Create a graph where every other cell is a node, neighbors have
        # edges, and edge weights are random
        graph = Graph()
        for r in range(0, rows, 2):
            for c in range(0, cols, 2):
                neighbors = [(r-2, c), (r+2, c), (r, c-2), (r, c+2)]
                for nr, nc in neighbors:
                    if 0 <= nr < rows and 0 <= nc < cols:
                        graph.addEdge((r, c), (nr, nc), random.random())

        # Use Kruskal's algorithm to obtain a minimum spanning tree, which
        # will have a random shape due to the random edge weights
        mst = graph.kruskal()
        
        # Create the grid and populate it with values based on the edges
        # from the mst obtained earlier
        grid = [[0 for c in range(cols)] for r in range(rows)]
        for r, c in mst.getNodes():
            for nr, nc in mst.getNeighbors((r, c)):
                grid[r][c] = 1
                grid[(r+nr)//2][(c+nc)//2] = 1
                grid[nr][nc] = 1
        return grid

    # Eller's algorithm for maze generation
    def eller(self, rows, cols):
        
        # Create a graph that will later become the maze
        trueRows, trueCols = rows, cols
        rows, cols = rows//2+1, cols//2+1
        G = Graph()

        # Keep track of which "set" each column of the first row belongs to,
        # and the name that the next set will be given
        rowSets = list(range(cols))
        nextSet = cols

        # Repeat the following procedure to generate all but the last row
        for row in range(rows-1):
            
            # Loop over each horizontal neighbor in the row and if they are
            # not in the same set, randomly decide whether or not to break the
            # wall between them
            for col in range(cols-1):
                setA = rowSets[col]
                setB = rowSets[col+1]
                if setA != setB and random.choice([True, True, True, False, False]):
                    
                    # If so, add the edge to the graph and ensure that all cols
                    # with those row sets are combined into one row set
                    G.addEdge((row, col), (row, col+1), 1)
                    for i in range(cols):
                        if rowSets[i] == setB:
                            rowSets[i] = setA
            
            # Identify the distinct row sets for the current row, create
            # the initially empty row sets for the next row, and loop over
            # each column in a random order
            uniqueSets = set(rowSets)
            newRowSets = [-1] * cols
            columns = list(range(cols))
            random.shuffle(columns)
            for col in columns:

                # If this column's row set does not already have a vertical
                # tunnel to the next row set, then add one. Otherwise, decide
                # randomly whether or not to add one.
                if rowSets[col] in uniqueSets or (random.choice([True, False])):
                    G.addEdge((row, col), (row+1, col), 1)
                    newRowSets[col] = rowSets[col]
                    uniqueSets.discard(rowSets[col])
            
            # Fill in the rest of the next row set that was not already touched
            # by the vertical tunnels from the current row by giving them new
            # row set labels
            for col in range(cols):
                if newRowSets[col] == -1:
                    newRowSets[col] = nextSet
                    nextSet += 1
            rowSets = newRowSets
        
        # Repeat the horizontal process once more for hte final node but without
        # choice, ensuring that any disjoint sets are combined
        for col in range(cols-1):
            setA = rowSets[col]
            setB = rowSets[col+1]
            if setA != setB:
                G.addEdge((rows-1, col), (rows-1, col+1), 1)
                for i in range(cols):
                    if rowSets[i] == setB:
                        rowSets[i] = setA

        # Create the grid and populate it with values based on the edges
        # from the graph created earlier
        grid = [[0 for c in range(trueRows)] for r in range(trueCols)]
        for r, c in G.getNodes():
            grid[2*r][2*c] = 1
            for nr, nc in G.getNeighbors((r, c)):
                grid[(2*r+2*nr)//2][(2*c+2*nc)//2] = 1
                grid[2*nr][2*nc] = 1
        return grid

    # Uses the Hunt & Kill algorithm to create a maze
    def hunt(self, rows, cols):

        # Create the initially empty grid
        grid = [[0 for c in range(cols)] for r in range(rows)]

        # Hunt procedure: looks for a cell (r0, c0) that is a wall but has an
        # open neighbor (r1, c1). Returns the tuple (r0, c0, r1, c1). If
        # this is the starting iteration, returns a random cell instead.
        def hunt(start=False):
            if start:
                r = 2 * random.randint(0, rows//2)
                c = 2 * random.randint(0, cols//2)
                return (r, c, r, c)

            # Loop over each cell in random order
            cells = [(r, c) for r in range(0, rows, 2) for c in range(0, cols, 2)]
            random.shuffle(cells)
            for r, c in cells:

                # If the cell is a wall but has an open neighbor, return the cell
                if grid[r][c] == 1: continue
                for nr, nc in [(r-2, c), (r+2, c), (r, c-2), (r, c+2)]:
                    if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 1:
                        return (nr, nc, r, c)

            # If no wall cells with open neighbors were found, return a tuple
            # of None values to indicate that the algorithm has terminated
            return (None, None, None, None)

        # The kill procedure: does a random walk starting at (r1, c1), where
        # the previously visited cell was (r0, c0).
        def kill(r0, c0, r1, c1):

            # Carve a path to the current cell from the previous one
            grid[(r0+r1)//2][(c0+c1)//2] = 1
            grid[r1][c1] = 1

            # List the neighbors that are walls
            neighbors = []
            for nr, nc in [(r1+2, c1), (r1+2, c1), (r1, c1-2), (r1, c1+2)]:
                if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 0:
                    neighbors.append((nr, nc))

            # If there is at least one such neighbor, continue the random walk
            # from that neighbor. Otherwise the kill phase ends.
            if len(neighbors) > 1:
                r2, c2 = random.choice(neighbors)
                kill(r1, c1, r2, c2,)

        # Repeat the hunt and kill procedures until the hunt phase ends
        r0, c0, r1, c1 = hunt(True)
        while r0 != None:
            kill(r0, c0, r1, c1)
            r0, c0, r1, c1 = hunt()
        return grid

    # The "islamic city" aproach to maze generation. Big idea: start with
    # a set of nodes that must all be reachable from one another, and randomly
    # add walls so long as they do not prevent all that from happening.
    def islamic(self, rows, cols):

        # Start with a designated seed node, and create a set of nodes that
        # must be reachable from one another, including all 4 corners but
        # otherwise random.
        seed = (0, 0)
        mustReach = {(0, 0), (0, rows-1), (cols-1, 0), (rows-1, cols-1)}
        while len(mustReach) < (rows*cols)//4:
            r, c = random.randint(0, rows-1), random.randint(0, cols-1)
            if (r, c) not in mustReach:
                mustReach.add((r, c))

        # Create the grid without any walls and then loop over all of the
        # possible wall positions in random order
        grid = [[1 for c in range(cols)] for r in range(rows)]
        walls = [(r, c) for r in range(rows) for c in range(cols) if (r, c) not in mustReach]
        random.shuffle(walls)
        while len(walls) > 0:

            # Extract the last 3 walls from the list and place them in the grid
            amt = min(3, len(walls))
            toPlace = []
            for i in range(amt): toPlace.append(walls.pop())
            for (r, c) in toPlace:
                grid[r][c] = 0

            # Extract the set of all nodes reachable from the seed node
            reachable = self.getConnected(rows, cols, grid, seed)

            # If any of the must reach nodes are not in that set, remove
            # the most recent batch of walls
            if len(mustReach-reachable) > 0:
                for (r, c) in toPlace:
                    grid[r][c] = 1

        # Find any pockets of nodes not reachable from the seed node and fill
        # them in with walls as well
        reachable = self.getConnected(rows, cols, grid, seed)
        for row in range(rows):
            for col in range(cols):
                if (row, col) not in reachable:
                    grid[row][col] = 0
        return grid
    
    # Create random mine-like cave by creating random rooms and then carving
    # tunnels between them with Kruskal's algorithm
    def miner(self, rows, cols):

        # Create the starting grid with all walls
        grid = [[0 for c in range(cols)] for r in range(rows)]

        numBombs = int(2.5 * rows ** 0.5)
        bombs = [(random.randint(0, rows-1), random.randint(0, cols-1), random.randint(3, int(rows ** 0.5)-1)) for i in range(numBombs)]
        for r, c, blastRadius in bombs:
            for row in range(rows):
                for col in range(cols):
                    d = ((r - row) ** 2 + (c - col) ** 2) ** 0.5
                    if d <= blastRadius:
                        grid[row][col] = 1

        ufs = {i: i for i in range(numBombs)}
        def find(node):
            if ufs[node] == node: return node
            result = find(ufs[node])
            ufs[node] = result
            return result
        def union(nodeA, nodeB):
            if random.random() > 0.5:
                ufs[nodeA] = nodeB
            else:
                ufs[nodeB] = nodeA
        
        def edgeWeight(i, j):
            dr = bombs[i][0] - bombs[j][0]
            dc = bombs[i][1] - bombs[j][1]
            d = (dr ** 2 + dc ** 2) ** 0.5
            shift = 1 + (random.random() - 0.5) / 2
            return d * shift, d
        edges = [(*edgeWeight(i, j), i, j) for i in range(numBombs) for j in range(i+1, numBombs)]
        edges.sort()
        for _, d, i, j in edges:
            r0, c0, b0 = bombs[i]
            r1, c1, b1 = bombs[j]
            repA, repB = find(i), find(j)
            if d < b0 + b1 - 2:
                union(repA, repB)
                continue
            if repA == repB and random.random(): continue
            union(repA, repB)
            theta = math.atan2(r1-r0, c1-c0)
            for i in range(int(d)+b0+b1-3):
                r = int(r0 + i * math.sin(theta))
                c = int(c0 + i * math.cos(theta))
                neighbors = [(r, c), (r-1, c), (r+1, c), (r, c-1), (r, c+1)]
                neighbors.pop(random.randint(1, 4))
                for nr, nc in neighbors:
                    if 0 <= nr < rows and 0 <= nc < cols:
                        grid[nr][nc] = 1
        return grid

    # Use random sine waves to generate a cave-like grid
    def sine(self, rows, cols):

        # Create the sinusodial functions
        sineFunctions = []
        for i in range(rows):
            sign = random.choice([-1, 1])
            period = random.random() ** 2
            amp = random.random() / period
            trig = random.choice([math.sin, math.cos])
            shift = random.random() * math.tau
            theta = random.random() * math.tau
            a, b = math.cos(theta), math.sin(theta)
            var = (lambda a, b: lambda r, c: r * a + c * b)(a, b)
            fn = (lambda sign, period, amp, trig, shift, var: lambda r, c: 
                sign * amp + trig(period * var(r, c) + shift))(sign, period, amp, trig, shift, var)
            sineFunctions.append(fn)

        # Create the grid by applying all of the wave functions and summing
        grid = [[sum(sineFn(r, c) for sineFn in sineFunctions) for c in range(cols)] for r in range(rows)]

        # Obtain the medain value
        values = [e for row in grid for e in row]
        values.sort()
        median = values[int(len(values)/2.2)]

        # Use values above the median as passages and below as walls
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == 0: grid[r][c] = grid[r+1][c]
                grid[r][c] = 0 if grid[r][c] < median else 1

        # Find the set of all nodes reachable from a random non-wall cell, and
        # start from scratch if the cave is then to small
        seed = random.choice([(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == 1])
        reachable = self.getConnected(rows, cols, grid, seed)
        if len(reachable) < (rows*cols)//3:
            return self.sine(rows, cols)

        # Fill in every pocket of nodes not reachable from that random cell
        for r in range(rows):
            for c in range(cols):
                if (r, c) not in reachable:
                    grid[r][c] = 0

        return grid

    # Use a cellular automata to generate a cave-like grid
    def automata(self, rows, cols):
        # Start with a random grid of 0s and 1s (~40% 0s, 60% 1s to start)
        grid = [[1 if random.random() > 0.4 else 0 for c in range(cols)] for r in range(rows)]

        # Key parameters controlling the rules of the automata
        iterations = 6
        minLife = 4
        maxLife = 3

        # Repeat the process a certain number of times (more = smoother)
        for i in range(iterations):

            # Create the next generation of the grid and loop over each
            # cell to see what its value is based on the current generation
            newGrid = [[0 for c in range(cols)] for r in range(rows)]
            for r in range(rows):
                for c in range(cols):

                    # Count the number of its 8 neighbors that are alive in
                    # the current generation
                    alive = 0
                    for dr in range(-1, 2):
                        for dc in range(-1, 2):
                            if dr == 0 and dc == 0: continue
                            nr, nc = r+dr, c+dc
                            if nr < 0 or nr >= rows or nc < 0 or nc >= cols: continue
                            if grid[nr][nc] == 0:
                                alive += 1

                    # Determine whether or not the cell is alive in the next
                    # generation based on the automata rules
                    if grid[r][c] == 1:
                        newGrid[r][c] = 1 if alive >= minLife else 0
                    else:
                        newGrid[r][c] = 1 if alive >= maxLife else 0
            
            # Replace the grid with the next generation grid
            grid = newGrid

        # Find the set of all nodes reachable from a random non-wall cell, and
        # start from scratch if the cave is then to small
        seed = random.choice([(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == 1])
        reachable = self.getConnected(rows, cols, grid, seed)
        if len(reachable) < (rows*cols)//3:
            return self.automata(rows, cols)

        # Fill in every pocket of nodes not reachable from that random cell
        for r in range(rows):
            for c in range(cols):
                if (r, c) not in reachable:
                    grid[r][c] = 0
        return grid

    # Use the Voronoi noise to generate a cave-like grid
    def voronoi(self, rows, cols):
        # Create the initially empty grid, a list of randomly placed seeds, and
        # a list that will store every value
        grid = [[0 for c in range(cols)] for r in range(rows)]
        seeds = [(random.randint(0, rows-1), random.randint(0, cols-1)) for i in range((rows+cols)//2)]
        values = []
        for r in range(rows):
            for c in range(cols):
                distances = []
                for sr, sc in seeds:
                    distances.append(1 / ((r - sr) ** 2 + (c - sc) ** 2 + 1))
                v = sum(distances) * (random.random() + 0.5)
                values.append(v)
                grid[r][c] = v

        # Obtain the median of the values and partition each cell as 0 or 1 based
        # on its position relative to the median
        values.sort()
        median = values[len(values)//2]
        for r in range(rows):
            for c in range(cols):
                grid[r][c] = 0 if grid[r][c] < median else 1

        # Make a copy of the grid with neighbor-rounded values
        grid = self.smoothGrid(rows, cols, grid)

        # Find the set of all nodes reachable from a random non-wall cell, and
        # start from scratch if the cave is then to small
        seed = random.choice([(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == 1])
        reachable = self.getConnected(rows, cols, grid, seed)
        if len(reachable) < (rows*cols)//3:
            return self.voronoi(rows, cols)

        # Fill in every pocket of nodes not reachable from that random cell
        for r in range(rows):
            for c in range(cols):
                if (r, c) not in reachable:
                    grid[r][c] = 0
        return grid

    # Use the diamond-square algorithm to generate a cave-like grid
    def diamondSquare(self, rows, cols):
        
        # Create the initial grid with randomized corners
        grid = [[0 for c in range(cols)] for r in range(rows)]
        grid[0][0] = random.random()
        grid[0][-1] = random.random()
        grid[-1][0] = random.random()
        grid[-1][-1] = random.random()

        # Square step procedure: sets the value of (r, c) based on the values
        # of the 4 cells defining a corner with (r, c) at its center, with
        # radius s. The amplitude defines how much random fuzz to add.
        def squareStep(r, c, s, amp):
            total, count = 0, 0
            if r-s >= 0 and c-s >= 0: 
                total += grid[r-s][c-s]
                count += 1
            if r-s >= 0 and c+s < cols: 
                total += grid[r-s][c+s]
                count += 1
            if r+s < rows and c-s >= 0: 
                total += grid[r+s][c-s]
                count += 1
            if r+s < rows and c+s < cols: 
                total += grid[r+s][c+s]
                count += 1
            grid[r][c] = total / count + (random.random() - 0.5) * amp

        # Diamond step procedure: sets the value of (r, c) based on the values
        # of the 4 cells defining a diamond with (r, c) at its center, with
        # radius s. The amplitude defines how much random fuzz to add.
        def diamondStep(r, c, s, amp):
            total, count = 0, 0
            if r-s >= 0:
                total += grid[r-s][c]
                count += 1
            if r+s < rows:
                total += grid[r+s][c]
                count += 1
            if c-s >= 0:
                total += grid[r][c-s]
                count += 1
            if c+s < cols:
                total += grid[r][c+s]
                count += 1
            grid[r][c] = total / count + (random.random() - 0.5) * amp

        # Repeat the procedure, halving the size each time until every
        # cell has been filled
        size = rows // 2
        amp = 1
        s = {(0, 0), (0, cols-1), (rows-1, 0), (rows-1, cols)}
        while True:
            half = size // 2
            if half < 1: break

            # Perform the square steps
            for r in range(half, rows, size):
                for c in range(half, cols, size):
                    squareStep(r, c, half, amp)

            # Perform the diamond steps
            for c in range(0, cols, half):
                startRow = half if c % 2 == 1 else 0
                for r in range(startRow, rows, half):
                    diamondStep(r, c, half, amp)

            size = half
            amp /= 2.5

        # Obtain the medain value
        values = [e for row in grid for e in row]
        values.sort()
        median = values[len(values)//2]

        # Use values above the median as passages and below as walls
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == 0: grid[r][c] = grid[r+1][c]
                grid[r][c] = 0 if grid[r][c] < median else 1

        # Make a copy of the grid with neighbor-rounded values
        grid = self.smoothGrid(rows, cols, grid)

        # Find the set of all nodes reachable from a random non-wall cell, and
        # start from scratch if the cave is then to small
        seed = random.choice([(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == 1])
        reachable = self.getConnected(rows, cols, grid, seed)
        if len(reachable) < (rows*cols)//3:
            return self.diamondSquare(rows, cols)

        # Fill in every pocket of nodes not reachable from that random cell
        for r in range(rows):
            for c in range(cols):
                if (r, c) not in reachable:
                    grid[r][c] = 0
        return grid

    # Use perlin noise to generate a cave-like grid
    def perlin(self, rows, cols):
        p = Perlin([(1, 0.25), (1, 0.2)])
        grid = [[1 if p(r/3, c/3) > 0.5 else 0 for c in range(cols)] for r in range(rows)]
        return grid

    # Use a modified DFS to create a pacman-like grid
    def pacrat(self, rows, cols):
        # Start by using DFS to create the top-left corner of the grid
        quadrant = self.dfs(rows//2+1, cols//2+1)

        # Create the real grid
        grid = [[0 for c in range(cols)] for r in range(rows)]

        # A funciton which fills in a certain cell of the grid, with horizontal
        # and vertical reflection symmetries
        def symmetricFill(r, c, val=1):
            grid[r][c] = val
            grid[r][-c-1] = val
            grid[-r-1][c] = val
            grid[-r-1][-c-1] = val

        # Add passages to every cell in the grid if they are passages in the
        # generated quadrant or in a set of designated open cells
        mustBeOpen = {(0, 0), (0, 1), (1, 0),
                      (rows//2, cols//4-1), (rows//4-1, cols//2),
                      (rows//2-1, cols//2), (rows//2, cols//2-1), #(rows//2, cols//2)
                      }
        for r in range(rows//2+1):
            for c in range(cols//2+1):
                symmetricFill(r, c, quadrant[r][c] | ((r, c) in mustBeOpen))

        # Randomly generate extra cycles in the graph
        extraCycles, cyclesAdded = rows/5, 0
        while cyclesAdded < extraCycles:
            
            # Randomly choose a node and its neighbor such that they are both
            # open but there is a wall between them
            r0, c0 = random.randint(0, rows-1), random.randint(0, cols-1)
            if grid[r0][c0] != 1: continue
            neighbors = [(r0, c0-2), (r0, c0+2), (r0-2, c0), (r0+2, c0)]
            r1, c1 = random.choice(neighbors)
            if not (0 <= r1 < rows and 0 <= c1 < cols and grid[r1][c1] == 1): continue
            r2, c2 = (r0+r1)//2, (c0+c1)//2
            if grid[r2][c2] == 1: continue

            # Add a passage between those two cells
            symmetricFill(r2, c2)
            cyclesAdded += 1
        return grid

    # Use a modified DFS to create a maze specifically for a pipe game
    def pipes(self, rows, cols):
        grid = self.dfs(2*rows, 2*cols)
        
        # Create the direction-set grid by compressing the 0s and 1s grid
        pipeGrid = [[set() for c in range(cols)] for r in range(rows)]
        dirs = [(-1, 0, "U"), (1, 0, "D"), (0, -1, "L"), (0, 1, "R")]
        for r in range(0, 2*rows, 2):
            for c in range(0, 2*cols, 2):
                
                # Identify which of the 4 directions should have an edge
                for dr, dc, d in dirs:
                    nr, nc = r+dr, c+dc
                    if 0 <= nr < 2*rows and 0 <= nc < 2*cols:
                        # Add an edge if there was one in the DFS maze, or
                        # randomly with a low-ish probability
                        if grid[nr][nc] == 1 or random.random() < 0.1:
                            pipeGrid[r//2][c//2].add(d)

                # Choose a number of times to rotate the set clockwise
                rotations = random.choice([0, 0, 0, 1, 2, 3])
                for i in range(rotations):
                    newSet = set()
                    if "U" in pipeGrid[r//2][c//2]: newSet.add("R")
                    if "R" in pipeGrid[r//2][c//2]: newSet.add("D")
                    if "D" in pipeGrid[r//2][c//2]: newSet.add("L")
                    if "L" in pipeGrid[r//2][c//2]: newSet.add("U")
                    pipeGrid[r//2][c//2] = newSet

        return pipeGrid

    # Uses backtrackers to create a killer sudoku grid
    def killer(self, rows, cols):
        # Create the initial grid with random numbers in each cell
        grid = [[["White", random.randint(1, rows)] for c in range(cols)] for r in range(rows)]

        # Create datastructures keeping track of the components, including
        # a set of all nodes added to a component, a dictionary mapping each
        # component name (a letter) to the set of all nodes in that component
        visited = set()
        components = dict()
        currentComponent = "A"
        maxCompSize = 6

        # Loop over all the nodes in random order (skip if already visited)
        nodes = [(r, c) for r in range(rows) for c in range(cols)]
        random.shuffle(nodes)
        for r, c in nodes:
            if (r, c) in visited: continue
            
            # Create a new component starting from this node
            visited.add((r, c))
            componentValues = {grid[r][c][1]}
            component = {(r, c)}

            while True:
                # Extract all the neighbors of the cells in the component already
                # if they are unvisited, on the board, and their value is not
                # already in the component. End the loop if there aren't any
                neighbors = []
                for row, col in component:
                    for nr, nc in [(row-1, col), (row+1, col), (row, col-1), (row, col+1)]:
                        if ((nr, nc) in visited or nr < 0 or nr >= rows or nc < 0 
                            or nc >= cols or grid[nr][nc][1] in componentValues): continue
                        neighbors.append((nr, nc))
                if neighbors == []: break

                # Pick a random neighbor and add it to the component, ending
                # the loop if this reaches the size limit for components
                row, col = random.choice(neighbors)
                component.add((row, col))
                componentValues.add((grid[row][col][1]))
                visited.add((row, col))
                if len(component) >= maxCompSize: break

            # Add the new component to the dictionary and increment the character
            components[currentComponent] = component
            currentComponent = chr(ord(currentComponent)+1)
        
        # Turn the components into a graph where each component is a node
        # and edges exist between components if any of the (r, c) within them
        # are neighbors in the grid
        G = Graph()
        for ordVal in range(ord("A"), ord(currentComponent)):
            comp = chr(ordVal)
            G.addNode(comp)
            compSet = components[comp]
            for node in G.getNodes():
                for r, c in components[node]:
                    for nr, nc in [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]:
                        if (nr, nc) in compSet:
                            G.addEdge(comp, node, 1)

        # The 4 colors used to color the grid, and a dictionary that will map
        # each component to a color
        colors = ["#D93232", "#F2811D", "#DED531", "#2ABFA4"]
        compColors = {comp: None for comp in components}

        # A backtracker to obtain a 4-coloring of the graph
        def graphColor(component):
            
            # Halt if every component has been colored, otherwise loop over
            # all 4 colors in random order
            if component not in components: return True
            random.shuffle(colors)
            for color in colors:

                # Skip this color if any of the neighboring components already
                # has this color
                allClear = True
                for neighbor in G.getNeighbors(component):
                    if compColors[neighbor] == color:
                        allClear = False
                        break
                if not allClear: continue

                # Otherwise, color the node, make a recursive call, return
                # if it succeeds otherwise unmake the move
                compColors[component] = color
                if graphColor(chr(ord(component)+1)): return True
                compColors[component] = None

            # If all 4 colors failed, backtrack
            return False

        # Start the backtracker from the first component
        graphColor("A")

        # Color the grid according to the component colors
        for component in components:
            color = compColors[component]
            for r, c in components[component]:
                grid[r][c][0] = color

        return grid

    # Uses a random walk to generate a sliding tile puzzle
    def sliding(self, rows, cols):
        # Create the grid in its solution state with checkerboarded colors,
        # an empty bottom-right corner, and row-major tile enumerations
        grid = [[None for c in range(cols)] for r in range(rows)]
        r0, c0 = (rows-1, cols-1)
        for r in range(rows):
            for c in range(cols):
                if (r, c) == (r0, c0):
                    color = "#222222"
                    value = ""
                else:
                    color = ["#BABFA8", "#D93232"][(r+c)%2]
                    value = r*cols + c + 1
                grid[r][c] = (color, value)

        # Maintain a set of all nodes that have been swapped at least once,
        # and continue swapping until the entire grid of all nodes have been
        # swapped at least once
        visited = {(rows-1, cols-1)}
        while len(visited) < rows * cols:

            # Obtain a list of all neighbors of the current empty cell
            neighbors = []
            for nr, nc in [(r0-1, c0), (r0+1, c0), (r0, c0-1), (r0, c0+1)]:
                if 0 <= nr < rows and 0 <= nc < cols:
                    neighbors.append((nr, nc))

            # Choose a random neighbor and swap with it
            r1, c1 = random.choice(neighbors)
            grid[r0][c0], grid[r1][c1] = grid[r1][c1], grid[r0][c0]
            visited.add((r1, c1))
            r0, c0 = r1, c1
        return grid

    # Uses Perlin noise to generate a Kami puzzle
    def kami(self, rows, cols):
        # Create the perlin function and the initial grid
        perlinFn = Perlin([(1, 0.25), (2, 0.15)])
        colors = ["#D93240", "#2B2D2A", "#0FA67B", "#D9C0A3"] * 2
        grid = [[(colors[0], "") for c in range(cols)] for r in range(rows)]

        # For each cell, calculate its perlin value, use a logistic function to
        # push it slightly away from the mean, and assign a color accordingly
        for r in range(rows):
            for c in range(cols):
                value = perlinFn(r/1.5, c/1.5)
                value = 1 / (1 + math.e ** (7 * (0.5 - value)))
                for i in range(len(colors)):
                    if value <= (i + 1) / len(colors):
                        grid[r][c] = (colors[i], "")
                        break
        return grid

    def keyPressed(self, event):
        if event.key == "Left":
            self.shiftAlgorithm(reverse=True)
        if event.key == "Right":
            self.shiftAlgorithm()
        elif event.key == "Space":
            self.makeMaze()
        elif event.key == "i":
            self.isometric = not self.isometric

    def getCellBounds(self, row, col, offset=False):
        cw = (self.gridSize / self.cols)
        ch = (self.gridSize / self.rows)
        x0 = (0 if offset else (self.width - self.gridSize)/2) + cw * col
        y0 = (0 if offset else (self.height - self.gridSize)/2) + ch * row
        x1 = x0 + cw
        y1 = y0 + ch
        return x0, y0, x1, y1

    def renderPipeCell(self, canvas, row, col):
        x0, y0, x1, y1 = self.getCellBounds(row, col)
        dx = (x1 - x0) / 3
        dy = (y1 - y0) / 3
        canvas.create_rectangle(x0, y0, x1, y1, fill="Black", width=0)
        canvas.create_rectangle(x0+dx, y0+dy, x1-dx, y1-dy, fill="White", width=0)
        if "U" in self.maze[row][col]:
            canvas.create_rectangle(x0+dx, y0, x1-dx, y1-dy, fill="White", width=0)
        if "D" in self.maze[row][col]:
            canvas.create_rectangle(x0+dx, y0+dy, x1-dx, y1, fill="White", width=0)
        if "L" in self.maze[row][col]:
            canvas.create_rectangle(x0, y0+dy, x1-dx, y1-dy, fill="White", width=0)
        if "R" in self.maze[row][col]:
            canvas.create_rectangle(x0+dx, y0+dy, x1, y1-dy, fill="White", width=0)
        canvas.create_rectangle(x0, y0, x1, y1, width=1)

    def renderColorCell(self, canvas, row, col):
        x0, y0, x1, y1 = self.getCellBounds(row, col)
        color, text = self.maze[row][col]
        canvas.create_rectangle(x0, y0, x1, y1, fill=color)
        if text != None:
            canvas.create_text((x0+x1)/2, (y0+y1)/2, text=text)

    def toIsometric(self, x, y, z=0):
        thetaX = math.radians(210)
        thetaY = math.radians(330)
        xi = x * math.cos(thetaX) + y * math.cos(thetaY)
        yi = x * math.sin(thetaX) + y * math.sin(thetaY)
        return self.width/2 + xi, self.height*0.25-yi

    def renderIsometricCell(self, canvas, row, col, h, topColor, leftColor, rightColor):
        x0, y0, x1, y1 = self.getCellBounds(row, col, True)
        ax, ay = self.toIsometric(x0, y0)
        bx, by = self.toIsometric(x0, y1)
        cx, cy = self.toIsometric(x1, y1)
        dx, dy = self.toIsometric(x1, y0)
        canvas.create_polygon(ax, ay-h, bx, by-h, cx, cy-h, dx, dy-h, fill=topColor, width=0)
        if h > 0:
            canvas.create_polygon(bx, by-h, cx, cy-h, cx, cy, bx, by, fill=leftColor, width=0)
            canvas.create_polygon(dx, dy-h, cx, cy-h, cx, cy, dx, dy, fill=rightColor, width=0)

    def renderMaze(self, canvas):
        for row in range(self.rows):
            for col in range(self.cols):
                if self.currentAlgorithm in self.pipeAlgos:
                    self.renderPipeCell(canvas, row, col)
                elif self.currentAlgorithm in self.colorAlgos:
                    self.renderColorCell(canvas, row, col)
                elif self.isometric:
                    if self.maze[row][col] == 1:
                        self.renderIsometricCell(canvas, row, col, 0, "#ffffff", None, None)
                    else:
                        self.renderIsometricCell(canvas, row, col, 8, "#202020", "#282828", "#303030")
                else:
                    x0, y0, x1, y1 = self.getCellBounds(row, col)
                    color = "White" if self.maze[row][col] == 1 else "Black"
                    canvas.create_rectangle(x0, y0, x1, y1, width=0, fill=color)

    def renderMazeName(self, canvas):
        x = self.width*0.5
        y = self.height*0.9
        canvas.create_text(x, y, text=self.currentAlgorithm, font="Futura 16 bold")

    def redrawAll(self, canvas):
        self.renderMaze(canvas)
        self.renderMazeName(canvas)

Mazes(width=800, height=800)

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
        self.menuWidth = (self.width - self.gridSize) / 3.25
        self.centerX = self.menuWidth + (self.width - self.menuWidth) / 2
        self.algorithms = {
            "Recursive Backtracking": (self.dfs, 31, 31), # Must be odd dims
            "Prim's Algorithm": (self.prim, 31, 31), # Must be odd dims
            "Kruskal's Algorithm": (self.kruskal, 31, 31), # Must be odd dims
            "Eller's Algorithm": (self.eller, 31, 31), # Must be odd dims
            "Hunt & Kill": (self.huntKill, 31, 31), # Must be odd dims
            "Wilson's Algorithm": (self.wilson, 31, 31), # Must be odd dims, ideally small-ish
            "Aldous-Broder": (self.aldous, 31, 31), # Must be odd dims, ideally small-ish
            "Binary Tree": (self.binary, 31, 31), # Must be odd dims
            "Sidewinder": (self.side, 31, 31), # Must be odd dims
            "Recursive Division": (self.division, 31, 31), # Must be odd dims
            "Blob Division": (self.blob, 51, 51), # Must be odd dims, ideally medium-large-ish
            "Weave DFS": (self.weaveDFS, 15, 15), # Can have any dims
            "Weave Kruskal": (self.weaveKruskal, 15, 15), # Can have any dims
            "Islamic City": (self.islamic, 31, 31), # Must be odd dims
            "Balloon Tunnel": (self.balloon, 100, 100), # Can have any dims, ideally large-ish
            "PVK Miner": (self.pvkMiner, 125, 125), # Can have any dims, ideally large-ish
            "Sine Waves": (self.sine, 75, 75), # Can have any dims, ideally medium-large-ish
            "Cellular Automata": (self.automata, 75, 75), # Ideally large dims
            "Voronoi Noise": (self.voronoi, 64, 64), # Can have any dims
            "Diamond Square": (self.diamondSquare, 129, 129), # Square power of 2 plus 1
            "Perlin Noise": (self.perlin, 125, 125), # Procedural generation :)
            "Pacman Grid": (self.pacrat, 31, 31), # Must have multiple of 4 - 1 dims
            "Twisted Pipes": (self.pipes, 7, 7), # Can have any dims
            "Regular Sudoku": (self.sudoku, 9, 9), # Perfect square, ideally small-ish
            "Killer Sudoku": (self.killer, 9, 9), # Square, ideally small-ish
            "Word Search": (self.wordSearch, 13, 13), # Size depends on word set used
            "Sliding Tile": (self.sliding, 5, 5), # Ideally small-ish
            "Kami Puzzle": (self.kami, 30, 30), # Ideally medium-small-ish
            }
        self.algorithmQueue = list(self.algorithms)
        self.currentAlgorithm = self.algorithmQueue[0]
        self.makeMaze()
        self.pipeWeaveAlgos = {"Twisted Pipes", "Weave DFS", "Weave Kruskal"}
        self.colorAlgos = {"Regular Sudoku", "Killer Sudoku", "Word Search", "Sliding Tile", "Kami Puzzle"}
        self.pipesWithBorders = {"Twisted Pipes"}
        self.colorWithoutBorders = {"Kami Puzzle"}
        self.isometric = False
        self.makeButtons()

    def makeButtons(self):
        bw = self.menuWidth
        bh = self.height / len(self.algorithmQueue)
        self.buttons = []
        for i in range(len(self.algorithmQueue)):
            algo = self.algorithmQueue[i]
            button = Button(bw/2, bh*(i+0.5), bw, bh, algo,
            "White", "Black", (lambda a: lambda: self.setAlgorithm(a))(algo))
            self.buttons.append(button)
        for button in self.buttons:
            button.hover(bw/2, bh/2)

    def setAlgorithm(self, algorithm):
        self.currentAlgorithm = algorithm
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

        # Invoke the recursive procedure from a random seed and return the grid
        sr, sc = 2*random.randint(0, rows//2), 2*random.randint(0, cols//2)
        recursiveBacktracker(sr, sc)
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
    def huntKill(self, rows, cols):

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

    def wilson(self, rows, cols):
        # Create the initially empty grid and the vector grid
        grid = [[0 for c in range(cols)] for r in range(rows)]
        vect = [[(0, 0) for c in range(cols)] for r in range(rows)]

        # Start with a random seed that is already a passage
        sr, sc = 2*random.randint(0, rows//2), 2*random.randint(0, cols//2)
        grid[sr][sc] = 1

        visited = {(sr, sc)}
        target = (rows//2 + 1) * (cols//2 + 1)



        while True:
            empty = [(r, c) for r in range(0, rows, 2) 
                                           for c in range(0, cols, 2)
                                           if grid[r][c] == 0]
            if len(empty) == 0: break
            sr, sc = random.choice(empty)
            r, c = sr, sc
            while grid[r][c] != 1:
                dirs = [(dr, dc) for dr, dc in [(0, -2), (0, 2), (-2, 0), (2, 0)]
                                 if 0 <= r+dr < rows and 0 <= c+dc < cols]
                dr, dc = random.choice(dirs)
                vect[r][c] = (dr, dc)
                r, c = r+dr, c+dc

            r, c = sr, sc
            while grid[r][c] != 1:
                dr, dc = vect[r][c]
                visited.add((r, c))
                grid[r][c] = 1
                grid[r+dr//2][c+dc//2] = 1
                r, c = r +dr, c+dc

        return grid
    
    # Uses the Aldous-Broder algorithm for maze generation
    def aldous(self, rows, cols):
        # Create the initially empty grid
        grid = [[0 for c in range(cols)] for r in range(rows)]

        # Does a stack-based random walk until the number of nodes visited
        # reached the target number (meaning every node has been hit)
        visited = set()
        target = (rows//2 + 1) * (cols//2 + 1)
        
        # Start the random walk from a random seed node
        r, c = 2*random.randint(0, rows//2), 2*random.randint(0, cols//2)
        while True:
            grid[r][c] = 1
            visited.add((r, c))
            if len(visited) == target: break

            # Choose one of the valid neighbors to continue the walk from,
            # carving a path to it if its also unvisited
            neighbors = [(r+dr, c+dc) for dr, dc in [(0, -2), (0, 2), (-2, 0), (2, 0)]
                                      if 0 <= r+dr < rows and 0 <= c+dc < cols]
            nr, nc = random.choice(neighbors)
            if (nr, nc) not in visited:
                grid[(r+nr)//2][(c+nc)//2] = 1
            r, c = nr, nc

        return grid

    # Uses the binary-tree algorithm for maze generation
    def binary(self, rows, cols):
        # Create the initially empty grid
        grid = [[0 for c in range(cols)] for r in range(rows)]

        # For every other cell, carve a passageway either up or left
        for r in range(0, rows, 2):
            for c in range(0, cols, 2):
                grid[r][c] = 1
                if (r == 0) and (c == 0): continue
                elif (r == 0) and (c > 0): grid[r][c-1] = 1
                elif (r > 0) and (c == 0): grid[r-1][c] = 1
                else:
                    if random.choice([True, False]):
                        grid[r][c-1] = 1
                    else:
                        grid[r-1][c] = 1

        return grid

    # Uses the sidewinder algorithm for maze generation
    def side(self, rows, cols):
        # Create the initially empty grid
        grid = [[0 for c in range(cols)] for r in range(rows)]

        # Carve out the entire top row
        for c in range(cols):
            grid[0][c] = 1

        # Carve every other cell
        for r in range(0, rows, 2):
            for c in range(0, cols, 2):
                grid[r][c] = 1

        # Repeat the carving sequences for each of the remaining rows
        for r in range(2, rows, 2):

            # Initialize the frequency, current column, and current row set
            freq = random.randint(2, int(rows ** 0.5) + 1)
            curCol = 0
            curRowSet = [0]
            while True:

                # If at the end, or with chance 1/4, end the current run and
                # carve upwards
                if curCol == cols-1 or random.random() < 1 / freq:

                    # Choose a random cell from the current row set to carve
                    # upwards from
                    c = random.choice(curRowSet)
                    grid[r-1][c] = 1

                    # If at the end of the row, end the current sequence
                    if curCol == cols-1: break

                    # Otherwise, re-randomize the frequency, increment the
                    # current column, and reset the current row set
                    freq = random.randint(2, int(rows ** 0.5) + 1)
                    curCol += 2
                    curRowSet = [curCol]

                # Otherwise, if not at the end, carve right (if possible)
                elif curCol < cols-1:
                    grid[r][curCol+1] = 1
                    curCol += 2
                    curRowSet.append(curCol)

        return grid

    # Uses the recursive-division algorithm for maze generation
    def division(self, rows, cols):

        # Creates the graph that represents every other cell in the maze
        G = Graph()
        for r in range(0, rows, 2):
            for c in range(0, cols, 2):
                G.addNode((r, c))

        # The recursive procedure which divides the range of nodes from (r0, c0)
        # to (r1, c1), inclusive of both, into two parts. Implemented via
        # a stack instead of directly with recursion
        stack = [(0, 0, rows-1, cols-1)]
        while len(stack) > 0:
            r0, c0, r1, c1 = stack.pop()
            if len(stack) > 100: break

            # If the range has width/height 1, just connect everything in the range
            if r0==r1 or c0==c1:
                for r2 in range(r0, r1+1, 2):
                    for c2 in range(c0, c1+1, 2):
                        for r3, c3 in [(r2-2, c2), (r2+2, c2), (r2, c2-2), (r2, c2+2)]:
                            if r0 <= r3 <= r1 and c0 <= c3 <= c1:
                                G.addEdge((r2, c2), (r3, c3), 1)
                continue

            # Randomly choose whether to divide horizontally or vertically,
            # with the probability biased depending on the range dims
            s = "HV" + ("V" if (r1 - r0) >= (c1 - c0) else "H")
            divisionType = random.choice(s)
            if divisionType == "H":

                # If horizontal, choose a bunch of columns to split the range
                # at randomly and then pick the one closest to the center of
                # the range
                cOptions = [random.randrange(c0, c1, 2) for i in range(7)]
                c = min(cOptions, key=lambda c: abs(c-(c0+c1)//2))

                # Add the two halves to be further divided into the stack
                stack.append((r0, c0, r1, c))
                stack.append((r0, c+2, r1, c1))

                # Pick a random row within the range and add a passage
                r = random.randrange(r0, r1+1, 2)
                G.addEdge((r, c), (r, c+2), 1)

            else:

                # If vertical, do the same but flip rows/cols
                rOptions = [random.randrange(r0, r1, 2) for i in range(7)]
                r = min(rOptions, key=lambda r: abs(r-(r0+r1)//2))
                stack.append((r0, c0, r, c1))
                stack.append((r+2, c0, r1, c1))
                c = random.randrange(c0, c1+1, 2)
                G.addEdge((r, c), (r+2, c), 1)

        # Create the grid from the nodes and edges of the graph
        grid = [[0 for c in range(cols)] for r in range(rows)]
        for node in G.getNodes():
            r0, c0 = node
            grid[r0][c0] = 1
            for neighbor in G.getNeighbors(node):
                r1, c1 = neighbor
                grid[(r0+r1)//2][(c0+c1)//2] = 1

        # Remove all the remaining buffer wall
        for r in range(1, rows, 2):
            for c in range(1, cols, 2):
                if grid[r-1][c] == 1 and grid[r+1][c] == 1 and grid[r][c-1] == 1 and grid[r][c+1] == 1:
                    grid[r][c] = 1
        return grid

    # Uses the blob-based recursive division algorithm for maze generation
    def blob(self, rows, cols):

        # Creates the graph that represents every other cell in the maze
        G = Graph()
        for r in range(0, rows, 2):
            for c in range(0, cols, 2):
                G.addNode((r, c))

        # The recursive procedure that divides a set of nodes into two parts
        # until the set is less than some size threshold
        maxRoomSize = 0.5 * (rows * cols) ** 0.5
        def divide(S):

            # If below that threshold, just connect everything in the set
            if len(S) <= maxRoomSize: 
                for r0, c0 in S:
                    for r1, c1 in [(r0-2, c0), (r0+2, c0), (r0, c0-2), (r0, c0+2)]:
                        if (r1, c1) not in S: continue 
                        G.addEdge((r0, c0), (r1, c1), 1)
                return

            # Pick two random seed nodes
            a, b = random.sample(S, k=2)
            A = {a}
            B = {b}
            C = {a, b}

            # Partition the nodes in the set into two components using a Prim's
            # algorithm-esque growth from the two seeds
            while len(C) < len(S):
                r0, c0 = random.sample(C, k=1)[0]
                for r1, c1 in [(r0-2, c0), (r0+2, c0), (r0, c0-2), (r0, c0+2)]:
                    if (r1, c1) not in S or (r1, c1) in C: continue
                    if (r0, c0) in A: A.add((r1, c1))
                    else: B.add((r1, c1))
                    C.add((r1, c1))

            # Try again if the division was too imbalanced
            if len(A) < maxRoomSize//3 or len(B) < maxRoomSize//3:
                divide(S)
                return

            # Identify all pairs of neighboring nodes on either side of the split
            border = []
            for r0, c0 in S:
                for r1, c1 in [(r0-2, c0), (r0+2, c0), (r0, c0-2), (r0, c0+2)]:
                    if (r1, c1) not in S: continue 
                    if ((r0, c0) in A) != ((r1, c1) in A):
                        border.append(((r0, c0), (r1, c1)))

            # Add one more edge connecting 2 nodes on opposite sides of the divide
            a, b = random.choice(border)
            G.addEdge(a, b, 1)

            # Divide each side of the split further
            divide(A)
            divide(B)

        # Run the recursive procedure starting with the set of all nodes
        divide(G.getNodes())

        # Create the grid from the nodes and edges of the graph
        grid = [[0 for c in range(cols)] for r in range(rows)]
        for node in G.getNodes():
            r0, c0 = node
            grid[r0][c0] = 1
            for neighbor in G.getNeighbors(node):
                r1, c1 = neighbor
                grid[(r0+r1)//2][(c0+c1)//2] = 1

        # Remove all the remaining buffer wall
        for r in range(1, rows, 2):
            for c in range(1, cols, 2):
                if grid[r-1][c] == 1 and grid[r+1][c] == 1 and grid[r][c-1] == 1 and grid[r][c+1] == 1:
                    grid[r][c] = 1
        return grid

    # The "islamic city" aproach to maze generation. Big idea: start with
    # a set of nodes that must all be reachable from one another, and randomly
    # add walls so long as they do not prevent all that from happening.
    def islamic(self, rows, cols):

        # Start with a designated seed node, and create a set of nodes that
        # must be reachable from one another, formed from every other row and
        # every other column
        seed = (0, 0)
        mustReach = {(r, c) for r in range(0, rows, 2) for c in range(0, cols, 2)}

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
    
    # Create random mine-like cave by creating randomly placed rooms (Voronoi
    # seed style) that grow outwards with a version of Prim's algorithm, and 
    # then carving tunnels between them with a version of Kruskal's algorithm
    def pvkMiner(self, rows, cols):

        # Create the starting grid with all walls
        grid = [[0 for c in range(cols)] for r in range(rows)]

        # Create the room centers in different parts of the cave
        numBombs = int(2 * rows ** 0.5)
        bombs = [(random.randint(1, rows-2), random.randint(1, cols-2)) for i in range(numBombs)]
        for r, c in bombs:
            grid[r][c] = 1
            component = {(r, c)}
            neighbors = [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]
            for i in range(random.randint((rows+cols)//3, 2*(rows+cols))):
                if neighbors == []: break
                row, col = neighbors.pop(random.randint(0, len(neighbors)-1))
                grid[row][col] = 1
                component.add((row, col))
                for nr, nc in [(row-1, col), (row+1, col), (row, col-1), (row, col+1)]:
                    if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 0:
                        neighbors.append((nr, nc))
                if neighbors == []: break

        # Implement the Kruskal UFS datastructure and utilities
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

        # Define a function that gets the edge weight of two rooms in the
        # cave based on the distance between their centers (plus some fuzz)
        def edgeWeight(i, j):
            dr = bombs[i][0] - bombs[j][0]
            dc = bombs[i][1] - bombs[j][1]
            d = (dr ** 2 + dc ** 2) ** 0.5
            shift = 1 + (random.random() - 0.5) / 2
            return d * shift, d

        # Create all the edges and loop over them in increasing order
        edges = [(*edgeWeight(i, j), i, j) for i in range(numBombs) for j in range(i+1, numBombs)]
        edges.sort()

        for _, d, i, j in edges:
            r0, c0 = bombs[i]
            r1, c1 = bombs[j]

            # Skip if the 2 rooms are already connected in the UFS
            repA, repB = find(i), find(j)
            if repA == repB: continue
            union(repA, repB)

            # Connect the two rooms by calculating the angle between them and
            # repeatedly removing blocks along that path, with some fuzz
            theta = math.atan2(r1-r0, c1-c0)
            for i in range(int(d)+3):
                r = int(r0 + i * math.sin(theta))
                c = int(c0 + i * math.cos(theta))
                neighbors = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]
                dr, dc = neighbors.pop(random.randint(1, 4))
                neighbors.append((-2*dr, -2*dc))
                for dr, dc in neighbors:
                    nr, nc = r+dr, c+dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        grid[nr][nc] = 1

        # Find every wall that is adjacent to a passage and make all of those
        # walls into passages to smoothing out the caves.
        adjacent = []
        for row in range(rows):
            for col in range(cols):
                if grid[row][col] == 1: continue
                for nr, nc in [(row-1, col), (row+1, col), (row, col-1), (row, col+1)]:
                    if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 1:
                        adjacent.append((row, col))
                        break
        for row, col in adjacent:
            grid[row][col] = 1

        return grid

    def balloon(self, rows, cols):
        
        # Create the starting grid with all walls
        grid = [[0 for c in range(cols)] for r in range(rows)]

        maxSize = (rows + cols) // 8
        minSize = maxSize // 4 + 1
        r0, c0, r1, c1 = minSize, minSize, rows-minSize-1, cols-minSize-1

        # Initialize the balloon list with one random balloon
        maxBalloons = int((rows + cols) ** 0.5)
        balloons = [(random.randint(r0, r1), random.randint(c0, c1), random.randint(minSize, maxSize))]
        maxIterations = 500
        iterations = 0

        # Repeatedly add more balloons until the max number of balloons or
        # iterations is reached, each time creating a new center and giving it
        # a radius such that it is tangent to existing circles
        while len(balloons) < maxBalloons:
            iterations += 1
            if iterations >= maxIterations: break
            row = random.randint(r0, r1)
            col = random.randint(c0, c1)
            minDistance = float("inf")
            for r, c, s in balloons:
                d = ((r - row) ** 2 + (c - col) ** 2) ** 0.5
                minDistance = min(minDistance, d - s)
            if minSize <= minDistance <= maxSize:
                balloons.append((row, col, minDistance))

        # For each balloon, carve out the ring of passages around it
        for r, c, s in balloons:
            for row in range(rows):
                for col in range(cols):
                    d = ((r - row) ** 2 + (c - col) ** 2) ** 0.5
                    if abs(d - s) <= 1.5:
                        grid[row][col] = 1

        # Backtracker which creates a DFS maze within each balloon, ensuring
        # that it touches the external ring exactly once
        def backtracker(r, c, visited, touchedRim):
            visited.add((r, c))
            neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            random.shuffle(neighbors)
            for dr, dc in neighbors:
                nr, nc = r+2*dr, c+2*dc
                if not (0 <= nr < rows and 0 <= nc < cols) or (nr, nc) in visited: continue
                adjacent = [(nr, nc), (nr-1, nc), (nr+1, nc), (nr, nc-1), (nr, nc+1)]
                touchingEdge = False
                for ar, ac in adjacent:
                    if 0 <= ar < rows and 0 <= ac < cols and grid[ar][ac] == 1:
                        touchingEdge = True
                        break
                if touchingEdge:
                    if touchedRim[0]: continue
                    touchedRim[0] = True
                    grid[(r+nr)//2][(c+nc)//2] = 1
                    grid[nr][nc] = 1
                    return
                grid[(r+nr)//2][(c+nc)//2] = 1
                grid[nr][nc] = 1
                backtracker(nr, nc, visited, touchedRim)

        # Invoke the backtracker once per ballooon
        for r, c, s in balloons:
            backtracker(r, c, set(), [False])

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
        seeds = [(random.randint(0, rows-1), random.randint(0, cols-1)) for i in range(rows)]
        values = []

        # For each cell in the grid, define its value as the sum of the
        # reciprocals of its distance from the 2 nearest seeds
        proximity = 2
        for r in range(rows):
            for c in range(cols):
                distances = []
                for sr, sc in seeds:
                    distances.append(1 / ((r - sr) ** 2 + (c - sc) ** 2 + 1))
                distances.sort()
                v = sum(distances[-proximity:]) #* (random.random() + 0.5)
                values.append(v)
                grid[r][c] = v

        # Obtain the median of the values and partition each cell as 0 or 1
        # based on its position relative to the median
        values.sort()
        median = values[len(values)//2]
        for r in range(rows):
            for c in range(cols):
                grid[r][c] = 0 if grid[r][c] > median else 1

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
        # p = Perlin([(1, 0.1)])
        p = Perlin([(1, 0.5), (2, 0.25), (3, 0.1)])
        grid = [[1 if p(r/3, c/3) > 0.5 else 0 for c in range(cols)] for r in range(rows)]
        return grid

    # Use a modified DFS to create a pacman-like grid
    def pacrat(self, rows, cols):
        # Start by using DFS to create the top-left corner of the grid
        quadrant = self.dfs(rows//2, cols//2)

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
        for r in range(rows//2):
            for c in range(cols//2):
                symmetricFill(r, c, quadrant[r][c])
        for r, c in mustBeOpen:
            symmetricFill(r, c, 1)

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

        # Create the initial grid of empty directions
        grid = [[set() for c in range(cols)] for r in range(rows)]
        
        # The recursive backtracking procedure which carves the maze
        visited = set()
        def backtracker(r, c):
            visited.add((r, c))
            moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            random.shuffle(moves)
            for dr, dc in moves:
                nr, nc = r+dr, c+dc
                if not (0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited):
                    continue
                if dr == -1:
                    grid[r][c].add("U")
                    grid[nr][nc].add("D")
                elif dr == 1:
                    grid[r][c].add("D")
                    grid[nr][nc].add("U")
                elif dc == -1:
                    grid[r][c].add("L")
                    grid[nr][nc].add("R")
                else:
                    grid[r][c].add("R")
                    grid[nr][nc].add("L")
                backtracker(nr, nc)
                
        # Start the backtracker at an arbitrary cell
        sr, sc = random.randint(0, rows-1), random.randint(0, cols-1)
        backtracker(sr, sc)

        # Randomly rotate each cell
        for r in range(rows):
            for c in range(cols):

                # Choose a number of times to rotate the set clockwise
                rotations = random.choice([0, 0, 0, 1, 2, 3])
                for i in range(rotations):

                    # Create a new set which contains each dir rotated once
                    newSet = set()
                    if "U" in grid[r][c]: newSet.add("R")
                    if "R" in grid[r][c]: newSet.add("D")
                    if "D" in grid[r][c]: newSet.add("L")
                    if "L" in grid[r][c]: newSet.add("U")
                    grid[r][c] = newSet

        return grid

    # Uses a modified DFS to generate a weave maze
    def weaveDFS(self, rows, cols):

        # Create the initial grid of empty directions
        grid = [[set() for c in range(cols)] for r in range(rows)]
        
        # The backtracking function which can carve in the normal directions
        # but can also tunnel underneath cells under certain conditions to
        # create a weave maze
        visited = set()
        def backtracker(r, c):
            visited.add((r, c))

            # Loops over the normal and weave moves, in random order
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            moves = [(dr, dc, moveType) for dr, dc in directions for moveType in "NW"]
            random.shuffle(moves)
            for dr, dc, moveType in moves:

                # If the move is normal...
                if moveType == "N":

                    # Skip if its out of bounds or already visited
                    nr, nc = r+dr, c+dc
                    if not (0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited):
                        continue

                    # Add the corresponding direction to the cell and the opposite
                    # direction to the neighbor, then make a recursive call from
                    # the neighbor
                    if dr == -1:
                        grid[r][c].add("U")
                        grid[nr][nc].add("D")
                    elif dr == 1:
                        grid[r][c].add("D")
                        grid[nr][nc].add("U")
                    elif dc == -1:
                        grid[r][c].add("L")
                        grid[nr][nc].add("R")
                    else:
                        grid[r][c].add("R")
                        grid[nr][nc].add("L")
                    backtracker(nr, nc)

                # If the move is a weave move...
                else:
                    r1, c1 = r+dr, c+dc
                    r2, c2 = r+2*dr, c+2*dc

                    # Skip if out of bounds, the endpoint of the tunnel is
                    # already visited, the midpoint of the tunnel is not
                    # visited, the tunnel is not perpendicular to the midpoint,
                    # or just randomly with a 30% chance
                    if not (0 <= r2 < rows and 0 <= c2 < cols):
                        continue
                    if (r1, c1) not in visited or (r2, c2) in visited:
                        continue
                    if dr != 0 and grid[r1][c1] != {"L", "R"}: continue
                    if dc != 0 and grid[r1][c1] != {"U", "D"}: continue
                    if random.random() > 0.7: continue

                    # Add the corresponding direction to the cell and the opposite
                    # direction to the neighbor, plus a tunnel in the cell in 
                    # between, then make a recursive call from the neighbor
                    if dr == -1:
                        grid[r][c].add("U")
                        grid[r1][c1].add("UD")
                        grid[r2][c2].add("D")
                    elif dr == 1:
                        grid[r][c].add("D")
                        grid[r1][c1].add("UD")
                        grid[r2][c2].add("U")
                    elif dc == -1:
                        grid[r][c].add("L")
                        grid[r1][c1].add("LR")
                        grid[r2][c2].add("R")
                    else:
                        grid[r][c].add("R")
                        grid[r1][c1].add("LR")
                        grid[r2][c2].add("L")
                    backtracker(r2, c2)

        # Start the backtracker at an arbitrary cell
        sr, sc = random.randint(0, rows-1), random.randint(0, cols-1)
        backtracker(sr, sc)
        return grid

    # Uses Kruskal's algorithm to generate a weave maze
    def weaveKruskal(self, rows, cols):
        # Create the initial grid of empty directions
        grid = [[set() for c in range(cols)] for r in range(rows)]

        # Implement the Kruskal UFS datastructure and utilities
        ufs = {(r, c): (r, c) for r in range(rows) for c in range(cols)}
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

        # Create the randomized locations of each crossing by looping over each
        # non-border cell in random order and adding it to the set of crossings
        # if it is not next to an existing crossing (+ some randomness)
        crossings = {}
        cells = [(r, c) for r in range(1, rows-1) for c in range(1, cols-1)]
        random.shuffle(cells)
        for r, c in cells:
            noNeighborCrossings = True
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                if (r+dr, c+dc) in crossings: noNeighborCrossings = False
                if (r+2*dr, c+2*dc) in crossings: noNeighborCrossings = False
            # If there are no neighboring crossings, add with high probability
            if noNeighborCrossings and random.random() > 0.1:
                crossings[r, c] = random.choice(["UD", "LR"])

        for r, c in crossings:
            # Update the UFS to account for the crossings
            repA, repB = find((r-1, c)), find((r+1, c))
            repC, repD = find((r, c-1)), find((r, c+1))
            union(repA, repB)
            union(repC, repD)

            # Update the grid to account for the crossing
            grid[r-1][c].add("D")
            grid[r+1][c].add("U")
            grid[r][c-1].add("R")
            grid[r][c+1].add("L")
            if crossings[r, c] == "UD":
                grid[r][c] |= {"L", "R", "UD"}
            else:
                grid[r][c] |= {"U", "D", "LR"}

        # Create all the remaining edges by skipping ones where either endpoint
        # is one of the crossings
        edges = []
        for r in range(rows):
            for c in range(cols):
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    r1, c1 = r+dr, c+dc
                    if (r, c) in crossings or (r1, c1) in crossings: continue
                    if 0 <= r1 < rows and 0 <= c1 < cols:
                        edges.append(((r, c), (r1, c1)))
        
        # Loop over the remaining edges in random order
        random.shuffle(edges)
        for a, b in edges:
            r0, c0 = a
            r1, c1 = b

            # Skip if the 2 rooms are already connected in the UFS
            repA, repB = find(a), find(b)
            if repA == repB: continue

            # If the move is normal...
            dr, dc = r1-r0, c1-c0
            if abs(dr) <= 1 and abs(dc) <= 1:

                # Add the corresponding direction to the cell and the opposite
                # direction to the neighbor
                if dr == -1:
                    grid[r0][c0].add("U")
                    grid[r1][c1].add("D")
                elif dr == 1:
                    grid[r0][c0].add("D")
                    grid[r1][c1].add("U")
                elif dc == -1:
                    grid[r0][c0].add("L")
                    grid[r1][c1].add("R")
                else:
                    grid[r0][c0].add("R")
                    grid[r1][c1].add("L")
                union(repA, repB)

        return grid

    # Uses backtrackers to create a Sudoku board
    def sudoku(self, rows, cols):
        # Create the initial grid
        grid = [[["White", 0] for c in range(cols)] for r in range(rows)]
        n = int(rows ** 0.5)

        # Create a list of each cell (accessed by the backtracker via an index
        # passed as a parameter so it knows which cell to look at next)
        cells = [(r, c) for r in range(rows) for c in range(cols)]

        # Finds the set of all numbers that can be placed at a certain cell
        def legalMoves(r, c):

            # Starts with every possible number
            legal = {i+1 for i in range(rows)}

            # Removes every number already in that row
            for col in range(cols):
                legal.discard(grid[r][col][1])

            # Removes every number already in that col
            for row in range(rows):
                legal.discard(grid[row][c][1])

            # Removes every number in that box
            r0, c0 = n * (r // n), n * (c // n)
            for row in range(r0, r0+n):
                for col in range(c0, c0+n):
                    legal.discard(grid[row][col][1])

            # Returns the remaining numbers in random order
            legal = list(legal)
            random.shuffle(legal)
            return legal

        # Recursive backtracker to place all of the digits, takes in the current
        # cell as an index i
        def backtracker(i):
            # Base case: when all of the cells have been filled
            if i >= len(cells): return True

            # Try each of the legal moves
            r, c = cells[i]
            for d in legalMoves(r, c):
                grid[r][c][1] = d
                if backtracker(i+1):
                    return True
            
            # If all the moves failed, undo and indicate falure
            grid[r][c][1] = 0
            return False

        # Run the backtracker from the first cell
        backtracker(0)

        # Color every other box a different color
        for r in range(rows):
            for c in range(cols):
                r0, c0 = n * (r // n), n * (c // n)
                if (r0 + c0) % 2 == 1:
                    grid[r][c][0] =  "#78C8F8"
        return grid

    # Uses backtrackers to create a killer sudoku grid
    def killer(self, rows, cols):
        # Create the initial grid with random numbers in each cell
        grid = self.sudoku(rows, cols)
        # grid = [[["White", random.randint(1, rows)] for c in range(cols)] for r in range(rows)]

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

    def wordSearch(self, rows, cols):
        # Create the initially empty grid, the list of words to use, and the
        # list of potential placements options for each word, in randomized
        # order for each word
        grid = [[["White", "*"] for c in range(cols)] for r in range(rows)]
        words = [
            "PYTHON", "RUBY", "RUST", "JAVA", "OCAML", "GOLANG", "TYPESCRIPT",
            "HASKELL", "LISP", "PERL", "PASCAL", "JAVASCRIPT", "OBJECTIVEC"]
        random.shuffle(words)
        moves = [(r, c, dr, dc) for r in range(rows)
                                for c in range(cols)
                                for dr in range(-1, 2)
                                for dc in range(-1, 2) if dr != 0 or dc != 0]

        # Stores all character placements (sperated by (-1, -1) so that they
        # can be undone
        stack = []

        # Returns True if a word can be placed with a certain move
        def canMakeMove(word, r0, c0, dr, dc):
            for i in range(len(word)):
                r = r0 + i * dr
                c = c0 + i * dc
                if not (0 <= r < rows and 0 <= c < cols): return False
                if grid[r][c][1] not in ("*", word[i]):
                    return False
            return True

        # Places a certain word in the grid
        def makeMove(word, r0, c0, dr, dc):
            stack.append((-1, -1))
            for i in range(len(word)):
                r = r0 + i * dr
                c = c0 + i * dc
                if grid[r][c][1] == "*":
                    grid[r][c][0] = "#D93232" if i == 0 else "#78C8F8"
                    grid[r][c][1] = word[i]
                    stack.append((r, c))

        # Unmakes the most recent move by popping off the stack until the
        # delimiter (-1, -1) is found
        def unmakeMove():
            while True:
                r, c = stack.pop()
                if (r, c) == (-1, -1): break
                grid[r][c][0] = "White"
                grid[r][c][1] = "*"

        # Uses recursive backtracking to place each word in the grid
        def backtracker(i):
            if i >= len(words): return True
            word = words[i]
            random.shuffle(moves)
            for r0, c0, dr, dc in moves:
                if canMakeMove(word, r0, c0, dr, dc):
                    makeMove(word, r0, c0, dr, dc)
                    if backtracker(i+1): return True
                    unmakeMove()
            return False

        backtracker(0)

        # Fill in all the remaining words
        for r in range(rows):
            for c in range(cols):
                if grid[r][c][1] != "*": continue
                grid[r][c][1] = chr(random.randint(ord("A"), ord("Z")))
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
                    value = None
                else:
                    color = ["#BABFA8", "#D93232"][(r+c)%2]
                    value = r*cols + c + 1
                grid[r][c] = [color, value]

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

    # Uses a modified version of the blob-based recursive division algorithm,
    # and the graph coloring backtracker, to generate a Kami puzzle
    def kami(self, rows, cols):

        # The recursive procedure that divides a set of nodes into two parts
        components = {}
        nextComponent = ["A"]
        def divide(S):

            # Stop dividing randomly, where the probability depends on the size
            # of the number of elements in S
            x = 9 * abs(len(S) / (rows * cols) - 0.01) ** 0.5
            if x == 0 or random.random() < 1 / x - 0.2:
                components[nextComponent[0]] = S
                nextComponent[0] = chr(ord(nextComponent[0]) + 1)
                return

            # Pick two random seed nodes
            a, b = random.sample(S, k=2)
            A = {a}
            B = {b}
            C = {a, b}

            # Partition the nodes in the set into two components using a Prim's
            # algorithm-esque growth from the two seeds
            while len(C) < len(S):
                r0, c0 = random.sample(C, k=1)[0]
                for r1, c1 in [(r0-1, c0), (r0+1, c0), (r0, c0-1), (r0, c0+1)]:
                    if (r1, c1) not in S or (r1, c1) in C: continue
                    if (r0, c0) in A: A.add((r1, c1))
                    else: B.add((r1, c1))
                    C.add((r1, c1))

            # Try again if the division was too imbalanced
            if not (0.1 <= len(A) / len(B) <= 10):
                divide(S)
                return

            # Divide each side of the split further
            divide(A)
            divide(B)

        # Run the recursive procedure starting with the set of all nodes
        S = {(r, c) for r in range(rows) for c in range(cols)}
        divide(S)

        # Create a graph where each component is a node and the edges are
        # between neighboring components
        G = Graph()
        for comp in components:
            G.addNode(comp)

            # Loop over each potentially neighboring component
            for other in components:
                if other == comp: continue
                foundNeighbor = False
                otherSet = components[other]

                # If any of the nodes of the current component have a neighbor
                # in the other component's set, add an edge between them
                for r, c in components[comp]:
                    for neighbor in [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]:
                        if neighbor in otherSet:
                            foundNeighbor = True
                            break
                    if foundNeighbor: break
                if foundNeighbor:
                    G.addEdge(comp, other, 1)



        # Create the initially empty grid, the 4 colors used to color the grid, 
        # and a dictionary that will map each component to a color
        colors = ["#D93240", "#2B2D2A", "#0FA67B", "#D9C0A3"]
        grid = [[[colors[0], None] for c in range(cols)] for r in range(rows)]
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

        # Color in each cell based on its component's color
        for comp in components:
            for r, c in components[comp]:
                grid[r][c][0] = compColors[comp]
                
        return grid

    def mousePressed(self, event):
        if event.x <= self.menuWidth:
            for button in self.buttons:
                button.hover(event.x, event.y)
                button.click(event.x, event.y)

    def keyPressed(self, event):
        if event.key == "Space":
            self.makeMaze()
        elif event.key == "i":
            self.isometric = not self.isometric

    def getCellBounds(self, row, col, offset=False):
        cw = (self.gridSize / self.cols)
        ch = (self.gridSize / self.rows)
        x0 = (0 if offset else self.centerX - self.gridSize/2) + cw * col
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
        if self.currentAlgorithm in self.pipesWithBorders:
            canvas.create_rectangle(x0, y0, x1, y1, width=1)

    def renderWeaveCell(self, canvas, row, col):
        if "LR" not in self.maze[row][col] and "UD" not in self.maze[row][col]:
            self.renderPipeCell(canvas, row, col)
            return
        x0, y0, x1, y1 = self.getCellBounds(row, col)
        dx = (x1 - x0) / 3
        dy = (y1 - y0) / 3
        canvas.create_rectangle(x0, y0, x1, y1, fill="Black", width=0)
        canvas.create_rectangle(x0+dx, y0, x1-dx, y1, fill="White", width=0)
        canvas.create_rectangle(x0, y0+dy, x1, y1-dy, fill="White", width=0)
        if "LR" in self.maze[row][col]:
            canvas.create_line(x0+dx, y0+dy, x1-dx, y0+dy)
            canvas.create_line(x0+dx, y1-dy, x1-dx, y1-dy)
        else:
            canvas.create_line(x0+dx, y0+dy, x0+dx, y1-dy)
            canvas.create_line(x1-dx, y0+dy, x1-dx, y1-dy)

    def renderColorCell(self, canvas, row, col):
        x0, y0, x1, y1 = self.getCellBounds(row, col)
        color, text = self.maze[row][col]
        w = 0 if self.currentAlgorithm in self.colorWithoutBorders else 1
        canvas.create_rectangle(x0, y0, x1, y1, fill=color, width=w)
        if text != None:
            canvas.create_text((x0+x1)/2, (y0+y1)/2, text=text)

    def toIsometric(self, x, y, z=0):
        thetaX = math.radians(213)
        thetaY = math.radians(327)
        xi = x * math.cos(thetaX) + y * math.cos(thetaY)
        yi = x * math.sin(thetaX) + y * math.sin(thetaY)
        return self.centerX + xi, self.height*0.25 - yi

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
                if self.currentAlgorithm in self.pipeWeaveAlgos:
                    self.renderWeaveCell(canvas, row, col)
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
        x = self.centerX
        y = self.height*0.9
        canvas.create_text(x, y, text=self.currentAlgorithm, font="Futura 16 bold")

    def redrawAll(self, canvas):
        for button in self.buttons:
            button.render(canvas, font="Futura 9 bold")
        self.renderMaze(canvas)
        self.renderMazeName(canvas)

Mazes(width=800, height=800)

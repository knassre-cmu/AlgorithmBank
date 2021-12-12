# AlgorithmBank
Code bank for various functions, algorithms, and classes

## Old Versions
Folder of the previous incarnate of the algorithm bank. This folder has a seperate README.md to explain its contents.

## Mazes
An animation which demonstrates a large breadth of maze/terrain generation algorithms. Contents of the folder include:
##### datastructures.py: a collection of classes that are useful for implementing various algorithms. Datastructures include:
- Red Black Trees (implement a dictionary interface)
- Priority Queues
- Graphs, with the following Graph algorithms included:
   - Extracting the set of all nodes connected to a node
   - Extracting a list of all connected components
   - Breadth First Search (BFS) pathfinding
   - Dijkstra's Algorithm pathfinding
   - A* Algorithm pathfinding
   - Kruskal's Algorithm for creating a minimum spanning tree
   - Edmond Karp #2 Algorithm for max flow
- Polygons, with the following computational geometry algorithms included:
   - Contianment of a point within a polygon
   - Line side test
   - Converting 2 points to standard form Ax + By = C
   - Convex hull (via Graham Scan)
   - Line intersection detection
   - Polygon overlap detection
   
##### Graphics.py: a tkinter graphics framework based on subclassing an App class to create a handler then subclassing a Mode class and invoking instances of it which can be toggled between by the App class. MVC conventions are expected, but not enforced. Includes the following:
- Methods that can be overwritten:
  - appStarted(app): called by App when it is initialized (expected to create and set modes)
  - appStarted(app): called by each mode when it is first entered
  - appStopped(app): called when the app stops running
  - timerFired(app): called every app.timerDelay milliseconds (100 by defaul)
  - mousePressed(app, event): called every time the mouse is pressed, with the location stored in the event parameter
  - mouseMoved(app, event): called every time the mouse is moved, with the location stored in the event parameter
  - mouseDragged(app, event): called every time the mouse is dragged, with the location stored in the event parameter
  - mouseReleased(app, event): called every time the mouse is released, with the location stored in the event parameter
  - keyPressed(app, event): called every time a non-modifier key is pressed, with the key stored in the event parameter
  - redrawAll(app, canvas): called after each event function is called
- Other methods provided:
  - app.setMode(name): takes in the name of a mode stored in the app as an attribute and sets the mode to that specific mode
  - app.getMousePos(): returns the current (x, y) location of the mouse
- Other classes provided:
  - Button: a button class that takes in the (cx, cy, w, h) dimensions, text, body color, text color, and onclick function. Provides click, hover, and render methods.
  - ScrollBar: a scroll bar class that takes in the (x0, y0, x1, y1) bounding box as well as an alias to a variable that it will adjust to indicate the current scroll value. Provides click, release, scroll, and render methods.
  
##### mazes.py: an animation created with Graphcis.py to highlight a plethora of maze/terrain generation algorithms, including:
- Maze Algorithms
  - Recursive Backtracking: uses Depth First Search (DFS) to create a maze
  - Prim's Algorithm: uses Prim's MST algorithm to create a maze
  - Kruskal's Algorithm: uses Kruskal's MST algorithm (from the Graph class) to create a maze
  - Eller's Algorithm: uses Eller's maze generation algorithm to create a maze
  - Hunt & Kill: uses the Hunt & Kill algorithm to create a maze
  - Islamic City: generates a maze by starting with no walls and a set of random nodes that must be reachable from one another, then adds all the remaining walls in random order unless they would prevent the key nodes from being reachable from one another (checked via floodfill). Air pockets are removed from the final grid via floodfill.
- Terrain Algorithms
  - Kruskal Miner: genereates a cave map by creating random circular rooms and then carving ragged tunnels between them using Kruskal's algorithm to determine which rooms to connect.
  - Sine Waves: generates a cave map by creating a random collection of sinusodial functions and applying them to each cell. Air pockets are removed from the final grid via floodfill.
  - Cellular Automata: generates a cave map by applying several iterations of a cellular automata to a random grid. Air pockets are removed from the final grid via floodfill.
  - Voronoi Noise: generates a cave map by randomly placing seeds and determining every cell's value based on the distances to every seed (plus some random fuzz). Air pockets are removed from the final grid via floodfill.
  - Diamond Square: generates a cave map via the Diamond-Square algorithm. Air pockets are removed from the final grid via floodfill.
  - Perlin Noise: genereates a map via Perlin noise
- Other Games/Puzzles
  - Pacman: uses a modified DFS to create a Pacman grid with extra cycles and symmetry
  - Killer Sudoku: uses several backtrackers to create a Killer Sudoku board with cage shapes using only 4 colors
  - Sliding Tile: uses random walk to create a sliding tile puzzle
  - Kami: uses Perlin noise to create a Kami puzzle
  
  
  
  
  

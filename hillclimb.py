import math

'''
Hill Climbing Algorithm

Input

initial: the initial state
heurisitc: an evaluation function that takes in a state and returns a number
moves: a function function that takes in a state and returns all adjacent states
n: returns the numebr of iterations to perform the algorithm

Output

state: returns the highest state achieved after n iterations
'''

# Main function of the hillclimbing algorithm. Generates all adjacent variants
# of the current state, calculates which has the best heuristic value, and
# makes that the next state. Repeats n times or until the initial is the best.
def HILLCLIMBER(initial,heuristic,moves,n):
    if n <= 0: return initial
    best = initial
    val = heuristic(initial)
    nextMoves = moves(initial)
    for move in nextMoves:
        h = heuristic(move)
        if h < val: 
            best = move
            val = h
    if best == initial: return initial
    return HILLCLIMBER(best,heuristic,moves,n-1)

# Takes in a spacing and returns a function that takes in an XY coordinate and
# returns the 8 adjacent locations where X and Y have been varied by that amount
def NEIGHBORFUNC(deltaXY):
    def NEIGHBORXY(location):
        out = []
        for dX in range(-1,2):
            for dY in range(-1,2):
                if dX == dY == 0: continue
                x = location[0] + dX*deltaXY
                y = location[1] + dY*deltaXY
                out.append((x,y))
        return out
    return NEIGHBORXY

# Returns if the x and y values in 2 tuples are almost equal
def TALEQ(u,v,e=10**-5):
    return ((u[0]-v[0])**2 + (u[1]-v[1])**2)**0.5 < e

# Test cases for HILLCLIMBER
def TESTHILLCLIMBER():
    print("Testing Hill Climber")
    f1 = lambda x: -math.e**-(x[0]**2+x[1]**2)
    f2 = lambda x: -math.e**-(x[0]**2+x[1]**2)-2*math.e**-((x[0]-1.7)**2+(x[1]-1.7)**2)
    m1 = NEIGHBORFUNC(0.1)
    m2 = NEIGHBORFUNC(0.2)
    l1 = (-1,-1)
    l2 = (-5,-3)
    l3 = (5,3)
    print(chr(10209)+" Test Case 1:","PASSED" if TALEQ(HILLCLIMBER(l1,f1,m1,5),(-0.5,-0.5)) else "FAILED")
    print(chr(10209)+" Test Case 2:","PASSED" if TALEQ(HILLCLIMBER(l1,f1,m1,10),(0,0)) else "FAILED")
    print(chr(10209)+" Test Case 3:","PASSED" if TALEQ(HILLCLIMBER(l2,f1,m2,10),(-3,-1)) else "FAILED")
    print(chr(10209)+" Test Case 4:","PASSED" if TALEQ(HILLCLIMBER(l3,f2,m2,20),(1.6,1.6)) else "FAILED")

if __name__ == "__main__": TESTHILLCLIMBER()
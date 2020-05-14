'''
Minimax + Alpha-Beta Pruning

Inputs

state: the initial state of the game
moves: a function that takes in a state and returns all moves that can be made
adjacent: a function that takes in a state and a move and returns a new state
created by making that move (does not modify state)
heuristic: a function that takes in a state and returns a numerical value that is
used for comparing the quality of different outcomes
maxDepth: the maximum depth of the game tree that should be searched

Outputs:

best move: the best move that can be made from the start state
'''

# Wrapper function for minimax + alpha-beta pruning. Performs top-level version.
def ABPMINIMAX(state,moves,adjacent,heuristic,maxDepth):
    candidates = moves(state)
    bestMove = None
    bestVal = float("-inf")
    alpha = float("-inf")
    beta = float("inf")
    for move in candidates:
        newState = adjacent(state,move) 
        val = ABPMINIMAXHELPER(newState,moves,adjacent,heuristic,maxDepth-1,False,alpha,beta)
        if val > bestVal:
            bestVal = val
            bestMove = move
        alpha = max(alpha, bestVal)
        if beta <= alpha: break
    return bestMove

# Main function for minimax + alpha-beta pruning. Recursively probes the game
# tree until maxDepth has been reached, and then returning the score at that
# leaf node. Alternates between maximizing and minimizing the score, and then
# prunes the game tree based on parameters alpha and beta.
def ABPMINIMAXHELPER(state,moves,adjacent,heuristic,depth,isMaximizing,alpha,beta):
    candidates = moves(state)
    if depth <= 0 or len(candidates) == 0: 
        h = heuristic(state)
        return h
    if isMaximizing:  
        bestVal = float("-inf")
        for move in candidates:
            newState = adjacent(state,move) 
            val = ABPMINIMAXHELPER(newState,moves,adjacent,heuristic,depth-1,False,alpha,beta)
            bestVal = max(bestVal, val)  
            alpha = max(alpha, bestVal)
            if beta <= alpha: break 
        return bestVal
    else: 
        bestVal = float("inf")
        for move in candidates:
            newState = adjacent(state,move) 
            val = ABPMINIMAXHELPER(newState,moves,adjacent,heuristic,depth-1,True,alpha,beta)
            bestVal = min(bestVal, val)
            beta = min(beta, bestVal)
            if beta <= alpha: break 
        return bestVal

def MOVES1(state):
    if state > 12: return []
    return [1,2,3]

def ADJACENT1(state,move):
    return 3*state+move

def HEURISTIC1(state):
    d = {13:3, 14:12, 15:8, 
         16:2, 17:4, 18:6, 
         19:14, 20:5, 21:2, 
         22:5, 23:3, 24:1, 
         25:4, 26:1, 27:9, 
         28:6, 29:5, 30:16, 
         31:42, 32:5, 33:3, 
         34:17, 35:18, 36:19, 
         37:2, 38:5, 39:3}
    return d[state]

def MOVES2(state):
    if state > 30: return []
    return [1,2]

def ADJACENT2(state,move):
    return 2*state+move

def HEURISTIC2(state):
    d = {31:3, 32:12, 33:8, 34:2, 35:4, 36:6, 37:14, 38:5,
         39:2, 40:5, 41:3, 42:1, 43:4, 44:1, 45:9, 46:6,
         47:5, 48:16, 49:42, 50:5, 51:3, 52:17, 53:18, 54:19,
         55:2, 56:5, 57:3, 58:16, 59:4, 60:6, 61:18, 62:3}
    return d[state]

def MOVES3(state):
    if state > 56: return []
    if state == 0: return [1,2,3,4,5,6,7,8]
    return [1,2]

def ADJACENT3(state,move):
    if state == 0: return state+move
    return 2*state+move+6

def HEURISTIC3(state):
    return (state**2)%23

# Test cases for ABPMINIMAX
def TESTABPMINIMAX():
    print(chr(10209)+" Test Case 1:","PASSED" if ABPMINIMAX(0,MOVES1,ADJACENT1,HEURISTIC1,4) == 1 else "FAILED")
    print(chr(10209)+" Test Case 2:","PASSED" if ABPMINIMAX(0,MOVES2,ADJACENT2,HEURISTIC2,6) == 2 else "FAILED")
    print(chr(10209)+" Test Case 3:","PASSED" if ABPMINIMAX(0,MOVES3,ADJACENT3,HEURISTIC3,7) == 7 else "FAILED")
    return 42

if __name__ == "__main__": TESTABPMINIMAX()

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

# Wrapper function for minimax + alpha-beta pruning. Picks the best move.
def ABPMINIMAX(state,moves,adjacent,heuristic,maxDepth):
    candidates = moves(state)
    return max(candidates,key=lambda m: ABPMINIMAXHELPER(adjacent(state,m),
    moves,adjacent,heuristic,maxDepth-1),False,float("-inf"),float("inf"))

# Main function for minimax + alpha-beta pruning. Recursively probes the game
# tree until maxDepth has been reached, and then returning the score at that
# leaf node. Alternates between maximizing and minimizing the score, and then
# prunes the game tree based on parameters alpha and beta.
def ABPMINIMAXHELPER(state,moves,adjacent,heuristic,depth,isMaximizing,alpha,beta):
    candidates = moves(state)
    if depth <= 0 or len(candidates) == 0: return heuristic(state)
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
            alpha = min(beta, bestVal)  
            if beta <= alpha: break 
        return bestVal

# Test cases for ABPMINIMAX
def TESTABPMINIMAX():
    return 42

if __name__ == "__main__": TESTABPMINIMAX()

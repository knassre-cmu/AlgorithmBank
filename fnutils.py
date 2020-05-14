'''
Memoize

Inputs

f: a function that takes in any number of hashable arguments

Outputs:

g: a deocrated version of f that caches results
'''

def MEMOIZE(f):
    memoDict = {}
    def g(*args):
        if args in memoDict: return memoDict[args]
        result = f(*args)
        memoDict[args] = result
        return result
    return g

'''
Reductor

Inputs

seed: a starting value for the reductor to apply to

Outputs

g: a decorator which takes in a function f and returns function which takes 
in a list L and returns f(f(f(seed,L[0]),L[1]),L[2])...
'''

# Main function for REDUCTOR.
def REDUCTOR(seed):
    def g(f):
        def h(L):
            if L == []: return seed
            return f(h(L[:-1]),L[-1])
        return h
    return g

# Demo function 1 for REDUCTOR. Returns a function that sums a list.
@REDUCTOR(0)
def f1(x,y): return x + y

# Demo function 2 for REDUCTOR. Returns a function that keeps evens from a list
@REDUCTOR([])
def f2(x,y): return x + [y] if y % 2 == 0 else x

# Demo function 3 for REDUCTOR. Returns a function that merges a 2D list into 1D
@REDUCTOR([])
def f3(x,y): return x + y

# Demo function 4 for REDUCTOR. Returns a function that squares the items from a list
@REDUCTOR([])
def f4(x,y): return x + [y**2]

# Test cases for REDUCTOR
def TESTREDUCTOR():
    print("Testing Reductor:")
    print(chr(10209)+" Test Case 1:","PASSED" if f1([1,2,3,4]) == 10 else "FAILED")
    print(chr(10209)+" Test Case 2:","PASSED" if f2([1,2,3,4]) == [2,4] else "FAILED")
    print(chr(10209)+" Test Case 3:","PASSED" if f3([[1],[2],[3,4]]) == [1,2,3,4] else "FAILED")
    print(chr(10209)+" Test Case 4:","PASSED" if f4([1,2,3,4]) == [1,4,9,16] else "FAILED")
    print()

if __name__ == "__main__": 
    TESTREDUCTOR()
import math

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

@MEMOIZE
def fibonacci(n):
    if n <= 1: return 1
    return fibonacci(n-1) + fibonacci(n-2)

# Test cases for MEMOIZE
def TESTMEMOIZE():
    print("Testing Memoize:")
    print(chr(10209)+" Test Case 1:","PASSED" if fibonacci(10) == 89 else "FAILED")
    print(chr(10209)+" Test Case 2:","PASSED" if fibonacci(25) == 121393 else "FAILED")
    print(chr(10209)+" Test Case 3:","PASSED" if fibonacci(50) == 20365011074 else "FAILED")
    print(chr(10209)+" Test Case 4:","PASSED" if fibonacci(75) == 3416454622906707 else "FAILED")
    print()

'''
Reductor

Inputs

seed: a starting value for the reductor to apply to

Outputs

g: a decorator which takes in a function f and returns a function which takes 
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

'''
Derivitive

Inputs

e: a nonzero number

Outputs

g: a decorator which takes in a mathematical function f(x) and returns a function
which approximates f'(x) using a tangent approximation with width e
'''

# Main function for DERIVATIVE
def DERIVATIVE(e):
    def g(f):
        def h(x):
            return (f(x+e)-f(x))/e
        return h
    return g

@DERIVATIVE(10**-8)
def f5(x):
    return x**2-2*x

@DERIVATIVE(10**-8)
def f6(x):
    return (2*x-x**2)**0.5

@DERIVATIVE(10**-8)
def f7(x):
    return math.log2(x)

# Test cases for DERIVATIVE
def TESTDERIVATIVE():
    print("Testing Derivative:")
    x1, s1 = 10, 18
    x2, s2 = 1, 0
    x3, s3 = 12, 1/(12*math.log(2,math.e))
    print(chr(10209)+" Test Case 1:","PASSED" if abs(f5(x1)-s1) < 10**-5 else "FAILED")
    print(chr(10209)+" Test Case 2:","PASSED" if abs(f6(x2)-s2) < 10**-5 else "FAILED")
    print(chr(10209)+" Test Case 3:","PASSED" if abs(f7(x3)-s3) < 10**-5 else "FAILED")
    print()

'''
Derivitive

Inputs

n: a positive number

Outputs

g: a decorator which takes in a mathematical function f(x) and returns a function
which takes in 2 x values (a,b) and returns the integral of f(x) from a to b
using n trapezoids
'''

def INTEGRAL(n):
    def g(f):
        def h(xA,xB):
            total = 0
            dX = (xB-xA)/n
            x1 = xA
            x2 = xA+dX
            for i in range(n):
                total += (f(x1)+f(x2))*dX/2
                x1 += dX
                x2 += dX
            return total
        return h
    return g

@INTEGRAL(500)
def f8(x):
    return x**3/10 - x**2/2 + 1

@INTEGRAL(500)
def f9(x):
    return 1+math.sin(x)

@INTEGRAL(500)
def f10(x):
    return math.e**-(x**2)

# Test cases for INTEGRAL
def TESTINTEGRAL():
    print("Testing Integral:")
    x1A, x1B, i1 = 0, 3, 0.525
    x2A, x2B, i2 = 0, math.tau, math.tau
    x3A, x3B, i3 = 0, 1, (math.pi/4)**0.5 * math.erf(1)
    print(chr(10209)+" Test Case 1:","PASSED" if abs(f8(x1A,x1B)-i1) < 10**-6 else "FAILED")
    print(chr(10209)+" Test Case 2:","PASSED" if abs(f9(x2A,x2B)-i2) < 10**-6 else "FAILED")
    print(chr(10209)+" Test Case 3:","PASSED" if abs(f10(x3A,x3B)-i3) < 10**-6 else "FAILED")
    print()

if __name__ == "__main__": 
    TESTMEMOIZE()
    TESTREDUCTOR()
    TESTDERIVATIVE()
    TESTINTEGRAL()
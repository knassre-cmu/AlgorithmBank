'''
Fibonacci Generator

Inputs

None

Outputs

g: a generator that will yield all the fibonacci numbers starting with 1, 1
'''

# Main function for FIBGEN. Yields the first 2 fibonacci numbers, and then
# with every stage updates the previous two fibonaci numbers and yields their sum
def FIBGEN():
    minusOne = 1
    minusTwo = 1
    yield 1
    yield 1
    while True:
        minusOne, minusTwo = minusOne+minusTwo, minusOne
        yield minusOne

# Test cases for FIBGEN
def TESTFIBGEN():
    print("Testing Fibonacci Generator:")
    F1 = []
    F2 = []
    F3 = []
    for f in FIBGEN():
        F1.append(f)
        if f > 10: break
    for f in FIBGEN():
        F2.append(f)
        if f > 50: break
    for f in FIBGEN():
        F3.append(f)
        if f > 90: break
    print(chr(10209)+" Test Case 1:","PASSED" if F1 == [1,1,2,3,5,8,13] else "FAILED")
    print(chr(10209)+" Test Case 2:","PASSED" if F2 == [1,1,2,3,5,8,13,21,34,55] else "FAILED")
    print(chr(10209)+" Test Case 3:","PASSED" if F3 == [1,1,2,3,5,8,13,21,34,55,89,144] else "FAILED")
    print()

'''
Infinite Range

Inputs

start: a number
step: a number

Outputs

g: a generator that will yield the range(start,∞,step)
'''

# Main function for IRANGE. Yields start, and then with every stage adds
# step and yields the new value.
def IRANGE(start,step):
    yield start
    while True:
        start += step
        yield start

# Test cases for IRANGE
def TESTIRANGE():
    print("Testing Infinite Range:")
    R1 = []
    R2 = []
    R3 = []
    for i in IRANGE(1,1):
        R1.append(i)
        if i > 9: break
    for i in IRANGE(2,2):
        R2.append(i)
        if i > 15: break
    for i in IRANGE(-1,0.25):
        R3.append(i)
        if i >= 1: break
    print(chr(10209)+" Test Case 1:","PASSED" if R1 == [1,2,3,4,5,6,7,8,9,10] else "FAILED")
    print(chr(10209)+" Test Case 2:","PASSED" if R2 == [2,4,6,8,10,12,14,16] else "FAILED")
    print(chr(10209)+" Test Case 3:","PASSED" if R3 == [-1,-0.75,-0.5,-0.25,0,0.25,0.5,0.75,1] else "FAILED")
    print()

'''
Infinite Iterator

Inputs

seed: a value
f: a function that takes in a value and returns a value

Outputs

g: a generator that will yield seed, f(seed), f(f(seed))...
'''

# Main function for IITER. Yields start, and then with every stage applies f
# start and yields the new value.
def IITER(seed,f):
    yield seed
    while True:
        seed = f(seed)
        yield seed

# Test cases for IITER
def TESTIITER():
    print("Testing Infinite Iterator:")
    I1 = []
    I2 = []
    I3 = []
    for i in IITER(10,lambda x: x/2):
        I1.append(i)
        if i < 1: break
    for i in IITER(1,lambda x: x*2):
        I2.append(i)
        if i > 100: break
    for i in IITER(2,lambda x: x**2):
        I3.append(i)
        if i > 1000: break
    print(chr(10209)+" Test Case 1:","PASSED" if I1 == [10,5,2.5,1.25,0.625] else "FAILED")
    print(chr(10209)+" Test Case 2:","PASSED" if I2 == [1,2,4,8,16,32,64,128] else "FAILED")
    print(chr(10209)+" Test Case 3:","PASSED" if I3 == [2,4,16,256,65536] else "FAILED")
    print()

'''
Flatten

Inputs

L: a possibly nested list

Outputs

l: a new list with all the elements of L but no nested lists
'''

# Main function for FLATTEN. If the input is not a list, returns it. If it is,
# returns a list of all the flattened elements.
def FLATTEN(L):
    if L == []: return []
    if not isinstance(L,list): return [L]
    return FLATTEN(L[0]) + FLATTEN(L[1:])

# Test cases for FLATTEN
def TESTFLATTEN():
    print("Testing Flatten:")
    L1 = [1,[2],[[3]]]
    L2 = [[[4,5,6]],[[7],8],[]]
    L3 = [[9,[10,[11,[12]]]],13]
    print(chr(10209)+" Test Case 1:","PASSED" if FLATTEN(L1) == [1,2,3] else "FAILED")
    print(chr(10209)+" Test Case 2:","PASSED" if FLATTEN(L2) == [4,5,6,7,8] else "FAILED")
    print(chr(10209)+" Test Case 3:","PASSED" if FLATTEN(L3) == [9,10,11,12,13] else "FAILED")
    print()

if __name__ == "__main__": 
    TESTFIBGEN()
    TESTIRANGE()
    TESTIITER()
    TESTFLATTEN()

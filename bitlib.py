'''
Get Bit

Inputs

n: a non-negative integer
k: a non-negative integer

Outputs

b: the kth bit of n
'''

# Main function of GETBIT. Shifts n to the right by n
def GETBIT(n,k):
    return (n >> k) & 1

# Test cases for GETBIT
def TESTGETBIT():
    print("Testing Get Bit:")
    print(chr(10209)+" Test Case 1:","PASSED" if GETBIT(3,1) == 1 else "FAILED")
    print(chr(10209)+" Test Case 2:","PASSED" if GETBIT(82,3) == 0 else "FAILED")
    print(chr(10209)+" Test Case 3:","PASSED" if GETBIT(2187,7) == 1 else "FAILED")
    print(chr(10209)+" Test Case 4:","PASSED" if GETBIT(16384,8) == 0 else "FAILED")
    print()

'''
Get Bit

Inputs

n: a non-negative integer
k: a non-negative integer
v: either 0 or 1

Outputs

b: n but with the kth bit flipped to v
'''

# Main function of SETBIT. Shifts n to the right by n
def SETBIT(n,k,v):
    if v == 0: return n & ~ (1 << k)
    if v == 1: return n | (1 << k)

# Test cases for SETBIT
def TESTSETBIT():
    print("Testing Set Bit:")
    print(chr(10209)+" Test Case 1:","PASSED" if SETBIT(3,1,0) == 1 else "FAILED")
    print(chr(10209)+" Test Case 2:","PASSED" if SETBIT(82,3,1) == 90 else "FAILED")
    print(chr(10209)+" Test Case 3:","PASSED" if SETBIT(2187,7,0) == 2059 else "FAILED")
    print(chr(10209)+" Test Case 4:","PASSED" if SETBIT(16384,8,1) == 16640 else "FAILED")
    print()

'''
Make Bitvector

Inputs

L: a rectangular 2D list of booleans

Outputs

B: an integer whose bits represent the True and False values
'''

# Main function of MAKEBV. Uses bitwise operators to set the appropriate bits
# to create an integer representation of a 2D array of booleans
def MAKEBV(L):
    B = 0
    for i in range(len(L)):
        for j in range(len(L[0])):
            B <<= 1
            if L[i][j]: B |= 1
    return B

# Test cases for MAKEBV
def TESTMAKEBV():
    print("Testing Make Bitvector:")
    B1 = [[True,False]]
    B2 = [[True,False],[True,False],[False,True]]
    B3 = [[True,False,True],[False,True,False],[True,False,True]]
    B4 = [[True,False,True],[True,True,True],[True,False,True],[True,True,False]]
    print(chr(10209)+" Test Case 1:","PASSED" if MAKEBV(B1) == 2 else "FAILED")
    print(chr(10209)+" Test Case 2:","PASSED" if MAKEBV(B2) == 41 else "FAILED")
    print(chr(10209)+" Test Case 3:","PASSED" if MAKEBV(B3) == 341 else "FAILED")
    print(chr(10209)+" Test Case 4:","PASSED" if MAKEBV(B4) == 3054 else "FAILED")
    print()

'''
Unmake Bitvector

Inputs

B: an integer whose bits represent the True and False values
u: a positive integer
v: a positive integer

Outputs

L: a rectangular 2D list of booleans with u rows and v cols
'''

# Main function of UNMAKEBV. Uses bitwise operators to find the appropriate bits
# of B and set the appropriate indices of the boolean array
def UNMAKEBV(B,u,v):
    L = [[False for c in range(v)] for r in range(u)]
    for r in range(u-1,-1,-1):
        for c in range(v-1,-1,-1):
            if (B & 1): L[r][c] = True
            B >>= 1
    return L

# Test cases for UNMAKEBV
def TESTUNMAKEBV():
    print("Testing Unmake Bitvector:")
    B1 = [[True,False]]
    B2 = [[True,False],[True,False],[False,True]]
    B3 = [[True,False,True],[False,True,False],[True,False,True]]
    B4 = [[True,False,True],[True,True,True],[True,False,True],[True,True,False]]
    print(chr(10209)+" Test Case 1:","PASSED" if UNMAKEBV(2,1,2) == B1 else "FAILED")
    print(chr(10209)+" Test Case 2:","PASSED" if UNMAKEBV(41,3,2) == B2 else "FAILED")
    print(chr(10209)+" Test Case 3:","PASSED" if UNMAKEBV(341,3,3) == B3 else "FAILED")
    print(chr(10209)+" Test Case 4:","PASSED" if UNMAKEBV(3054,4,3) == B4 else "FAILED")
    print()

'''
Bitvector Manipulation

A - Index Bitvector

Inputs

B: an integer whose bits represent the True and False values
u: a positive integer representing the number of rows
v: a positive integer representing the number of cols
r: the row being checked
c: the col being checked

Outputs

b: Bit of B at index (r,c)

B - Set Bitvector

Inputs

B: an integer whose bits represent the True and False values
u: a positive integer representing the number of rows
v: a positive integer representing the number of cols
r: the row being checked
c: the col being checked
k: either 0 or 1

Outputs

B: a new integer that represents B but with bit (r,c) flipped to k
'''

# Main function for INDEXBV. Returns the bit at (r,c)
def INDEXBV(B,u,v,r,c):
    return (B >> (v-c-1+v*(u-r-1))) & 1

# Main funtion for SETBV. Returns a newbitvector with (r,c) flipped to k
def SETBV(B,u,v,r,c,k):
    if k == 0: return B & ~ (k << (v-c-1+v*(u-r))) # FIX DIS
    if k == 1: B | (k << (v-c-1+v*(u-r)))


# Test cases for INDEXBV and SETBV
def TESTINDEXSETBV():
    print("Testing Index Bitvector:")
    B1 = MAKEBV([[True,False]])
    B2 = MAKEBV([[True,False],[True,False],[False,True]])
    B3 = MAKEBV([[True,False,True],[False,True,False],[True,False,True]])
    B4 = MAKEBV([[True,False,True],[True,True,True],[True,False,True],[True,True,False]])
    print(chr(10209)+" Test Case 1:","PASSED" if INDEXBV(B1,1,2,0,0) == 1 else "FAILED")
    print(chr(10209)+" Test Case 2:","PASSED" if INDEXBV(B2,3,2,1,1) == 0 else "FAILED")
    print(chr(10209)+" Test Case 3:","PASSED" if INDEXBV(B3,3,3,2,0) == 1 else "FAILED")
    print(chr(10209)+" Test Case 4:","PASSED" if INDEXBV(B4,4,3,3,2) == 0 else "FAILED")
    print()
    print("Testing Set Bitvector:")
    r = 1
    c = 2
    for i in range(r):
        for j in range(c):
            print(INDEXBV(B1,r,c,i,j),end=" ")
        print()
    print("\/")
    for i in range(r):
        for j in range(c):
            print(INDEXBV(SETBV(B1,r,c,0,0,0),r,c,i,j),end=" ")
        print()
    # print(chr(10209)+" Test Case 1:","PASSED" if SETBV(B1,1,2,0,0,0) == 1 else "FAILED")
    # print(chr(10209)+" Test Case 2:","PASSED" if SETBV(B2,3,2,1,1,1) == 0 else "FAILED")
    # print(chr(10209)+" Test Case 3:","PASSED" if SETBV(B3,3,3,2,0,0) == 1 else "FAILED")
    # print(chr(10209)+" Test Case 4:","PASSED" if SETBV(B4,4,3,3,2,1) == 0 else "FAILED")
    print()

if __name__ == "__main__":
    TESTGETBIT()
    TESTSETBIT()
    TESTMAKEBV()
    TESTUNMAKEBV()
    TESTINDEXSETBV()
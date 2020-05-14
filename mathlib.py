from fnutils import *

'''
nPrimes

Inputs

n: a positive integer

Outputs

primeList: a list of primes less than n, obtained using the Sieve of Eratosthenes
'''

# Main function for nPrimes algorithm. Starts with p = 2 and creates a mark array.
# Increments p until that index in the mark array has not been marked. Adds p to
# the primeList, then marks p(p), p(p+1), (p+2)... and repeats the process
def NPRIMES(n):
    markDict = [0]*n
    primeList = []
    p = 2
    while True:
        while p < n and markDict[p]: p += 1
        if p >= n: return primeList
        markDict[p] = 1
        primeList.append(p)
        for i in range(p**2,n,p):
            markDict[i] = 1

def TESTNPRIMES():
    print("Testing nPrimes:")
    P1 = [2,3,5,7]
    P2 = [2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97]
    P3 = [2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97,
          101,103,107,109,113,127,131,137,139,149,151,157,163,167,173,179,181,191,
          193,197,199,211,223,227,229,233,239,241,251,257,263,269,271,277,281,283,
          293,307,311,313,317,331,337,347,349,353,359,367,373,379,383,389,397,401,
          409,419,421,431,433,439,443,449,457,461,463,467,479,487,491,499,503,509,
          521,523,541,547,557,563,569,571,577,587,593,599,601,607,613,617,619,631,
          641,643,647,653,659,661,673,677,683,691,701,709,719,727,733,739,743,751,
          757,761,769,773,787,797,809,811,821,823,827,829,839,853,857,859,863,877,
          881,883,887,907,911,919,929,937,941,947,953,967,971,977,983,991,997]
    print(chr(10209)+" Test Case 1:","PASSED" if NPRIMES(10) == P1 else "FAILED")
    print(chr(10209)+" Test Case 2:","PASSED" if NPRIMES(100) == P2 else "FAILED")
    print(chr(10209)+" Test Case 3:","PASSED" if NPRIMES(1000) == P3 else "FAILED")
    print()

'''
Prime Factors

Inputs

n: a positive integer greater than 1

Outputs

fDict: a dictionary of the prime factors of n and how many times they divide n
'''

def PRIMEFACTORS(n):
    fDict = {}
    p = 2
    while n != 1:
        while n % p == 0 and n != 1:
            fDict[p] = fDict.get(p,0) + 1
            n //= p
        p += 1
    return fDict

def TESTPRIMEFACTORS():
    print("Testing Prime Factors:")
    PF1 = {2:2, 3:2, 5:1, 17:1}
    PF2 = {2:8, 5:3, 7:2}
    PF3 = {2:10, 3:5, 5:2, 7:1, 11:1}
    PF4 = {2:10, 3:4, 5:2, 7:1, 11:1, 13:1, 17:1, 19:1}
    print(chr(10209)+" Test Case 1:","PASSED" if PRIMEFACTORS(3060) == PF1 else "FAILED")
    print(chr(10209)+" Test Case 2:","PASSED" if PRIMEFACTORS(1568000) == PF2 else "FAILED")
    print(chr(10209)+" Test Case 3:","PASSED" if PRIMEFACTORS(479001600) == PF3 else "FAILED")
    print(chr(10209)+" Test Case 4:","PASSED" if PRIMEFACTORS(670442572800) == PF4 else "FAILED")
    print()

'''
GCD

a: a positive integer
b: a positive integer less than or equal to u

Outputs

d: the greatest common divisor of u and v
'''

# Main funciton for GCD. Finds the Greatest common divisor of a and b using
# Euclid's algorithm
def GCD(a,b):
    if a == 0: return b
    return GCD(b%a,a)

# Test cases for GCD
def TESTGCD():
    print("Testing GCD:")
    print(chr(10209)+" Test Case 1:","PASSED" if GCD(6798,612) == 6 else "FAILED")
    print(chr(10209)+" Test Case 2:","PASSED" if GCD(123456,54321) == 3 else "FAILED")
    print(chr(10209)+" Test Case 3:","PASSED" if GCD(3628800,7168) == 1792 else "FAILED")
    print(chr(10209)+" Test Case 4:","PASSED" if GCD(670442572800,479001600) == 159667200 else "FAILED")
    print()

'''
Catalan Numbers

Inputs

n: a positive integer

Outputs

c: the nth Catalan number (using memoization)
'''

# Main function for CATALAN using the following rules:
# • CATALAN(0) = 1
# • CATALAN(n) = CATALAN(0)*CATALAN(n-1) + CATALAN(1)*CATALAN(n-2) + ...
# (uses memoization decorator to cache results of recursive calls)
@MEMOIZE
def CATALAN(n):
    if n == 0: return 1
    total = 0
    for i in range(n):
        total += CATALAN(i) * CATALAN(n-i-1)
    return total

# Test cases for CATALAN
def TESTCATALAN():
    print("Testing Catalan:")
    print(chr(10209)+" Test Case 1:","PASSED" if CATALAN(1) == 1 else "FAILED")
    print(chr(10209)+" Test Case 2:","PASSED" if CATALAN(2) == 2 else "FAILED")
    print(chr(10209)+" Test Case 3:","PASSED" if CATALAN(4) == 14 else "FAILED")
    print(chr(10209)+" Test Case 4:","PASSED" if CATALAN(8) == 1430 else "FAILED")
    print(chr(10209)+" Test Case 5:","PASSED" if CATALAN(16) == 35357670 else "FAILED")
    print()

'''
Bell Numbers

Inputs

n: a positive integer

Outputs

b: the number of ways to partition a set of size n (using memoization)
'''

# Main function for BELL using the following rule: BELL(n) = sum (k=0->n) STERLING(n,k)
def BELL(n):
    total = 0
    for k in range(n+1):
        total += STERLING(n,k)
    return total

# Calculates STERLING(n,k), the number of ways of splitting n items into k groups
@MEMOIZE
def STERLING(n,k):
    if n < k or k <= 0: return 0
    if n <= 0 or n == k: return 1
    return k * STERLING(n-1,k) + STERLING(n-1,k-1)

# Test cases for BELL
def TESTBELL():
    print("Testing Bell:")
    print(chr(10209)+" Test Case 1:","PASSED" if BELL(1) == 1 else "FAILED")
    print(chr(10209)+" Test Case 2:","PASSED" if BELL(2) == 2 else "FAILED")
    print(chr(10209)+" Test Case 3:","PASSED" if BELL(4) == 15 else "FAILED")
    print(chr(10209)+" Test Case 4:","PASSED" if BELL(8) == 4140 else "FAILED")
    print(chr(10209)+" Test Case 5:","PASSED" if BELL(16) == 10480142147 else "FAILED")
    print()

if __name__ == "__main__": 
    TESTNPRIMES()
    TESTPRIMEFACTORS()
    TESTGCD()
    TESTCATALAN()
    TESTBELL()
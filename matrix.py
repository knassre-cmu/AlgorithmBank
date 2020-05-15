'''
MATRIX Class

Takes in 2 positive integers (rows,cols) and represents a mathematical matrix of 
size rows x cols. Can perform matrix addition and subtraction using + and - 
operators. Can perform scalar and matrix multiplication using * operator. 
Use .transpose() to return the transposed matrix. Can index into a matrix 
using list indexing notation but in the form [r,c]. Can take in initial values
for the matrix using the keyword argument "values." Matrices can be printed
and will display in the console with the appropriate number of lines, and with
appropriate spacing.
'''

class MATRIX(object):
    def __init__(self,rows,cols,**kwargs):
        self.rows = rows
        self.cols = cols
        self.arr = [[0 for c in range(self.cols)] for r in range(self.rows)]
        if "values" in kwargs: self.setValues(kwargs["values"])

    def setValues(self,values):
        assert(isinstance(values,list))
        assert(isinstance(values[0],list))
        assert(len(values) == self.rows)
        for r in range(self.rows): assert(len(values[r]) == self.cols)
        for r in range(self.rows):
            for c in range(self.cols):
                self.arr[r][c] = values[r][c]

    def transpose(self):
        out = MATRIX(self.cols,self.rows)
        zipped = list(zip(*self.arr))
        for r in range(self.cols):
            for c in range(self.rows):
                out[r,c] = zipped[r][c]
        return out

    def __repr__(self):
        s = ""
        colSizes = {c:1 for c in range(self.cols)}
        for r in range(self.rows):
            for c in range(self.cols):
                colSizes[c] = max(colSizes[c],len(str(self.arr[r][c])))
        for r in range(self.rows):
            s += "| "
            for c in range(self.cols):
                n = str(self.arr[r][c])
                while len(n) < colSizes[c]+1: n += " "
                s += n
            s += "|\n"
        return s.strip()

    def __eq__(self,other):
        assert(isinstance(other,MATRIX))
        return self.arr == other.arr

    def __add__(self,other):
        assert(isinstance(other,MATRIX))
        assert(self.rows == other.rows)
        assert(self.cols == other.cols)
        out = MATRIX(self.rows,self.cols)
        for r in range(self.rows):
            for c in range(self.cols):
                out[r,c] = self[r,c] + other[r,c]
        return out

    def __sub__(self,other):
        assert(isinstance(other,MATRIX))
        assert(self.rows == other.rows)
        assert(self.cols == other.cols)
        out = MATRIX(self.rows,self.cols)
        for r in range(self.rows):
            for c in range(self.cols):
                out[r,c] = self[r,c] - other[r,c]
        return out

    def __mul__(self,other):
        assert(isinstance(other,(int,float,MATRIX)))
        if isinstance(other,(int,float)):
            out = MATRIX(self.rows,self.cols)
            for r in range(self.rows):
                for c in range(self.cols):
                    out[r,c] = other * self[r,c]
        else:
            assert(self.cols == other.rows)
            out = MATRIX(self.rows,other.cols)
            trn = other.transpose()
            for r in range(self.rows):
                for c in range(other.cols):
                    out[r,c] = sum([i[0]*i[1] for i in zip(self.arr[r], trn.arr[c])])
        return out

    def __rmul__(self,other):
        assert(isinstance(other,(int,float)))
        out = MATRIX(self.rows,self.cols)
        for r in range(self.rows):
            for c in range(self.cols):
                out[r,c] = other * self[r,c]
        return out

    def __getitem__(self,key):
        assert(isinstance(key,tuple))
        assert(len(key) == 2)
        assert(key[0] in range(self.rows))
        assert(key[1] in range(self.cols))
        return self.arr[key[0]][key[1]]

    def __setitem__(self,key,value):
        assert(isinstance(key,tuple))
        assert(len(key) == 2)
        assert(key[0] in range(self.rows))
        assert(key[1] in range(self.cols))
        self.arr[key[0]][key[1]] = value
        
# Test cases for MATRIX
def TESTMATRIX():
    print("Testing MATRIX Class:")
    M1 = MATRIX(2,3)
    M2 = MATRIX(3,3,values=[[1,12,31],[4,5,6],[72,8,9]])
    M3 = MATRIX(3,3,values=[[1,2,3],[4,5,6],[7,8,9]])
    M4 = MATRIX(3,3,values=[[2,14,34],[8,10,12],[79,16,18]])
    M5 = MATRIX(3,3,values=[[0,10,28],[0,0,0],[65,0,0]])
    M6 = MATRIX(2,3,values=[[1,2,3],[4,5,6]])
    M7 = MATRIX(3,2,values=[[1,4],[2,5],[3,6]])
    M8 = MATRIX(2,2,values=[[14,32],[32,77]])
    M9 = MATRIX(3,3,values=[[3,6,9],[12,15,18],[21,24,27]])
    M1[1,1] = 2
    S2 = "| 1  12 31 |\n| 4  5  6  |\n| 72 8  9  |"
    print(chr(10209)+" Test Case 1:","PASSED" if str(M2) == S2 else "FAILED")
    print(chr(10209)+" Test Case 2:","PASSED" if M1[1,1] == 2 else "FAILED")
    print(chr(10209)+" Test Case 3:","PASSED" if M2 + M3 == M7 else "FAILED")
    print(chr(10209)+" Test Case 4:","PASSED" if M2 - M3 == M5 else "FAILED")
    print(chr(10209)+" Test Case 5:","PASSED" if M6.transpose() == M7 and M7.transpose() == M6 else "FAILED")
    print(chr(10209)+" Test Case 6:","PASSED" if M3*3 == M9 else "FAILED")
    print(chr(10209)+" Test Case 7:","PASSED" if M6*M7 == M8 else "FAILED")
    print()

if __name__ == '__main__': TESTMATRIX()
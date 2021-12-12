import math

'''
PRIORITYQUEUE Class

Represents a worklist where items can be added in with different priorities and
removed in order of least priority to greatest. Use .add() to add an item
and its priority, and .rem() to remove and return the item with the least priority.
Use .peak() to return the element with the least priority without removing it.
Priority queues can be printed and will display all elements in heap layer order.
Can take the length of a priority queue.
'''

class PRIORITYQUEUE(object):
    def __init__(self):
        self.arr = [(float("-inf"),float("-inf"))]

    def isEmpty(self):
        return len(self.arr) < 2

    def __len__(self):
        return len(self.arr)-1

    def __repr__(self):
        layers = []
        layer = []
        i = 1
        prev = 0
        gap = 1
        while i <= len(self):
            layer.append(str(self.arr[i][0]))
            i += 1
            if i == len(self)+1 or i == 2 or i-prev == gap:
                layers.append(" • ".join(layer))
                layer = []
                prev = i
                gap *= 2
        return "< " + " | ".join(layers) + " >"

    # Adds an element to the priority queue
    def add(self,elem,prior):
        pos = len(self.arr)
        self.arr.append((elem,prior))
        self.siftUp(pos)

    # Removes and returns the element on top
    def rem(self):
        assert(not self.isEmpty())
        elem, weight = self.arr[1]
        last = self.arr.pop()
        if len(self.arr) > 1:
            self.arr[1] = last
            self.siftDown(1)
        return elem, weight

    # Returns the element on top without removing it
    def peak(self):
        assert(not self.isEmpty())
        elem, weight = self.arr[1]
        return elem, weight

    # Restores ordering after an element has been added
    def siftUp(self,pos):
        if self.arr[pos][1] < self.arr[pos//2][1]:
            self.arr[pos], self.arr[pos//2] = self.arr[pos//2], self.arr[pos]
            self.siftUp(pos//2)

    # Restores ordering after an element has been removed
    def siftDown(self,pos):
        left = 2*pos
        right = 2*pos + 1
        if left >= len(self.arr): return
        elif self.arr[pos][1] < self.arr[left][1] and (right >= len(self.arr) or 
        self.arr[pos][1] < self.arr[right][1]): return
        elif right >= len(self.arr) or self.arr[left][1] < self.arr[right][1]:
            self.arr[pos], self.arr[left] = self.arr[left], self.arr[pos]
            self.siftDown(left)
        else:
            self.arr[pos], self.arr[right] = self.arr[right], self.arr[pos]
            self.siftDown(right)

# Test cases for PRIORITYQUEUE Class
def TESTPQ():
    print("Testing PRIORITYQUEUE Class:")
    PQ1 = PRIORITYQUEUE()
    for i in range(65,83): PQ1.add(chr(i),i)
    S1 = "< A | B • C | D • E • F • G | H • I • J • K • L • M • N • O | P • Q • R >"
    S2 = "< B | D • C | H • E • F • G | P • I • J • K • L • M • N • O | R • Q >"
    S3 = "< C | D • F | H • E • L • G | P • I • J • K • Q • M • N • O | R >"
    S4 = "< D | E • F | H • J • L • G | P • I • R • K • Q • M • N • O >"
    print(chr(10209)+" Test Case 1:","PASSED" if str(PQ1) == S1 else "FAILED")
    print(chr(10209)+" Test Case 2:","PASSED" if PQ1.rem() == ("A",65) else "FAILED")
    print(chr(10209)+" Test Case 3:","PASSED" if str(PQ1) == S2 else "FAILED")
    print(chr(10209)+" Test Case 4:","PASSED" if PQ1.rem() == ("B",66) else "FAILED")
    print(chr(10209)+" Test Case 5:","PASSED" if str(PQ1) == S3 else "FAILED")
    print(chr(10209)+" Test Case 6:","PASSED" if PQ1.rem() == ("C",67) else "FAILED")
    print(chr(10209)+" Test Case 7:","PASSED" if str(PQ1) == S4 else "FAILED")
    print(chr(10209)+" Test Case 8:","PASSED" if PQ1.peak() == ("D",68) else "FAILED")
    print()

if __name__ == '__main__': TESTPQ()

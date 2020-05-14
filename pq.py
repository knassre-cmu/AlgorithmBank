'''
PRIORITYQUEUE Class

Represents a worklist where items can be added in with different priorities and
removed in order of least priority to greatest. Use .add() to add an item
and its priority, and .rem() to remove and return the item with the least priority.
'''

class PRIORITYQUEUE(object):
    def __init__(self):
        self.arr = [(float("-inf"),float("-inf"))]

    def isEmpty(self):
        return len(self.arr) < 2

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
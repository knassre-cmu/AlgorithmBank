'''
QUEUE Class

Represents a worklist where items are removed in the order in which they were added.
Use .enq() to add an item to the end of the queue and .deq() to remove and reutrn
the item at the front of the queue. Use .peak() to return the element at the 
front of the queue without removing it. Queues can be printed and will display
the elements in order.
'''

class LINK(object):
    def __init__(self,val):
        self.val = val
        self.nxt = None

class QUEUE(object):
    def __init__(self):
        self.front = None
        self.back = None

    # Returns a string version of the queue for debugging
    def __repr__(self):
        qList = []
        link = self.front
        while link != None:
            qList.append(str(link.val))
            link = link.nxt
        return "< " + " << ".join(qList) + " >"

    def isEmpty(self):
        return self.front == None

    # Adds an element to the back of the queue
    def enq(self,val):
        link = LINK(val)
        if self.isEmpty():
            self.front = link
            self.back = link
        else:
            self.back.nxt = link
            self.back = link
    
    # Returns and removes the element at the front of the queue
    def deq(self):
        assert(not self.isEmpty())
        val = self.front.val
        self.front = self.front.nxt
        return val
    
    # Returns the element at the front of the queue without removing it
    def peak(self):
        assert(not self.isEmpty())
        val = self.front.val
        return val

# Test cases for QUEUE Class
def TESTQ():
    print("Testing QUEUE Class:")
    Q1 = QUEUE()
    for i in range(100,113): Q1.enq(chr(i))
    S1 = "< d << e << f << g << h << i << j << k << l << m << n << o << p >"
    S2 = "< e << f << g << h << i << j << k << l << m << n << o << p >"
    S3 = "< f << g << h << i << j << k << l << m << n << o << p >"
    S4 = "< g << h << i << j << k << l << m << n << o << p >"
    print(chr(10209)+" Test Case 1:","PASSED" if str(Q1) == S1 else "FAILED")
    print(chr(10209)+" Test Case 2:","PASSED" if Q1.deq() == "d" else "FAILED")
    print(chr(10209)+" Test Case 3:","PASSED" if str(Q1) == S2 else "FAILED")
    print(chr(10209)+" Test Case 4:","PASSED" if Q1.deq() == "e" else "FAILED")
    print(chr(10209)+" Test Case 5:","PASSED" if str(Q1) == S3 else "FAILED")
    print(chr(10209)+" Test Case 6:","PASSED" if Q1.deq() == "f" else "FAILED")
    print(chr(10209)+" Test Case 7:","PASSED" if str(Q1) == S4 else "FAILED")
    print(chr(10209)+" Test Case 8:","PASSED" if Q1.peak() == "g" else "FAILED")
    print()

if __name__ == '__main__': TESTQ()
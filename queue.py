'''
QUEUE Class

Represents a worklist where items are removed in the order in which they were added.
Use .enq() to add an item to the end of the queue and .deq() to remove and reutrn
the item at the front of the queue.
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
            qList.append(link.val)
            link = link.nxt
        return "<" + str(qList)[1:-1] + ">"

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
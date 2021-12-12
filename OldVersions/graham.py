from graph import *
import math

'''
Graham Scan

Inputs

graph: an instance of the class Graph
locX: a function that takes in a node and returns its x-location on the plane
locY: a function that takes in a node and returns its y-location on the plane

Outputs

cList: a list of nodes that form a convex hull around all nodes in graph
'''

# Wrapper function for Graham Scan. Initializes the polar array and scan stack.
def GSCAN(graph,locX,locY):
    bPoint = BPOINT(graph,locX,locY)
    polArr = [bPoint] + POLARRAY(graph,bPoint,locX,locY)
    if len(polArr) < 3: return None
    stack = [polArr[i] for i in range(3)]
    GSCANHELPER(graph,stack,polArr,locX,locY,3)
    return stack

# Returns the bottom left point
def BPOINT(graph,locX,locY):
    bPoint = None
    bX = float("inf")
    bY = float("inf")
    for node in graph.nodes:
        x, y = locX(node), locY(node)
        if y < bY or (y == bY and x < bX):
            bPoint = node
            bX, bY = x, y
    return bPoint

# Returns the polar angle between 2 points
def POLARANG(point1,point2,locX,locY):
    dX = locX(point1)-locX(point2)
    dY = locY(point1)-locY(point2)
    theta = math.atan2(dX,dY)
    if theta < 0: theta += 2*math.pi
    return theta

# Sorts the nodes by polar angle relative to bPoint. In cases of ties, keeps the furthest point.
def POLARRAY(graph,bPoint,locX,locY):
    polArr = []
    for node in graph.nodes:
        if node == bPoint: continue
        i = len(polArr)
        while i > 0:
            theta1 = POLARANG(bPoint,polArr[i-1],locX,locY)
            theta2 = POLARANG(bPoint,node,locX,locY)
            dist1 = (locX(polArr[i-1])-locX(bPoint))**2 + (locY(polArr[i-1])-locY(bPoint))**2
            dist2 = (locX(node)-locX(bPoint))**2 + (locY(node)-locY(bPoint))**2
            if theta1 > theta2 or (theta1 == theta2 and dist1 < dist2): break
            i -= 1
        polArr.insert(i,node)
    i = 0
    while i < len(polArr)-1:
            theta1 = POLARANG(bPoint,polArr[i],locX,locY)
            theta2 = POLARANG(bPoint,polArr[i+1],locX,locY)
            if theta1 == theta2: polArr.pop(i)
            else: i+= 1
    return polArr

# Main function for Graham Scan. Recursively loops through the nodes in the polar array
# and pops nodes off the stack until a counterclockwise formation is formed, then pushes
# the node onto the stack.
def GSCANHELPER(graph,stack,polArr,locX,locY,i):
    if i >= len(polArr): return
    node = polArr[i]
    top = stack[-1]
    mid = stack[-2]
    theta2 = POLARANG(mid,top,locX,locY)
    theta1 = POLARANG(top,node,locX,locY)
    if theta1 > theta2: 
        stack.pop()
        GSCANHELPER(graph,stack,polArr,locX,locY,i)
    else: 
        stack.append(node)
        GSCANHELPER(graph,stack,polArr,locX,locY,i+1)

# Test cases for GSCAN
def TESTGSCAN():
    print("Testing Graham Scan:")
    G1 = GRAPH()
    G2 = GRAPH()
    G3 = GRAPH()
    G4 = GRAPH()
    N1 = [(1,1),(1,3),(5,0),(4,2),(2,0),(3,1),(4,5),(6,3)]
    N2 = [(10,10),(11,20),(12,15),(13,14),(14,17),(15,11),(16,19),(17,18),(18,13),(19,16),(20,12)]
    N3 = [(36,66),(34,41),(96,56),(92,81),(60,78),(44,54),(87,16),(80,51),(71,85),(66,91),(20,89),
          (20,69),(27,84),(25,64),(2,48),(39,21)]
    for n in N1: G1.addNode(n)
    for n in N2: G2.addNode(n)
    for n in N3: G3.addNode(n)
    C1 = [(2,0),(5,0),(6,3),(4,5),(1,3),(1,1)]
    C2 = [(10,10),(20,12),(19,16),(17,18),(16,19),(11,20)]
    C3 = [(87,16),(96,56),(92,81),(66,91),(39,21)]
    lfx = lambda n: n[0]
    lfy = lambda n: n[1]
    print(chr(10209)+" Test Case 1:","PASSED" if GSCAN(G1,lfx,lfy) == C1 else "FAILED")
    print(chr(10209)+" Test Case 2:","PASSED" if GSCAN(G2,lfx,lfy) == C2 else "FAILED")
    print(chr(10209)+" Test Case 3:","PASSED" if GSCAN(G3,lfx,lfy) == C3 else "FAILED")
    print()

if __name__ == "__main__": TESTGSCAN()

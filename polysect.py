'''
Polygonal Intersection

Inputs

polyA: a list of (x,y) tuples representing the points of the first polygon
polyB: a list of (x,y) tuples representing the points of the second polygon

Outputs:

intersects: True if polyA and polyB intersect, false otherwise
'''

# Returns True if for 3 collinear points (p1, p2, p3), p2 is on the segment p1-p3
def ONLINSEG(p1,p2,p3):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    return min(x1,x3) <= x2 <= max(x1,x3) and min(y1,y3) <= y2 <= max(y1,y3)

# Returns "CLOCK" if p1-p2-p3 form a clockwise trio, "COUNT" if they form a
# counterclockwise trio, and "COLIN" if they form a colinear trio
def ORIENT(p1,p2,p3):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    v = (y2-y1)*(x3-x2)-(x2-x1)*(y3-y2)
    if v > 0: return "CLOCK"
    if v < 0: return "COUNT"
    return "COLIN"

# Returns True if the line segments p1-p2 and p3-p4 intersect
def INTERSECT(p1,p2,p3,p4):
    tripples = [(p1,p2,p3),(p1,p2,p4),(p3,p4,p1),(p3,p4,p2)]
    orientations = []
    for tripple in tripples:
        orientations.append(ORIENT(*tripple))
    if orientations[0] != orientations[1] and orientations[2] != orientations[3]: return True
    for i in range(4):
        if orientations[i] == "COLIN":
            return ONLINSEG(tripples[i][0],tripples[i][2],tripples[i][1])
    return False

# Checks if a point is inside of a polygon by counting the number of times the line
# starting from that point and extending it indefinately intersects with the polygon.
# If the number is odd, the point is inside the polygon. If not, it is even.
# Takes in an optional parameter extension to determine how far to extend the line.
# Also cases on point-based intersections by extending slightly further into the polygon.
def INSIDE(point,polygon,extension=10**6):
    count = 0
    p1 = point
    p2 = (point[0]+extension,point[1])
    for i in range(len(polygon)):
        p3 = polygon[i]
        p4 = polygon[(i+1)%len(polygon)]
        if INTERSECT(p1,p2,p3,p4):
            amt = 1
            o1 = ORIENT(p1,p3,p2)
            o2 = ORIENT(p1,p4,p2)
            o3 = ORIENT(p3,p1,p4)
            o4 = ORIENT(p3,p2,p4)
            if o1 == "COLIN": continue
            if o2 == "COLIN": continue
            if o3 == "COLIN": continue
            if o4 == "COLIN": continue
            count += 1
    for p in polygon:
        if ORIENT(p1,p,p2) == "COLIN":
            newPoint = (p[0]+0.01,p[1])
            if ONLINSEG(p1,p,p2) and INSIDE(newPoint,polygon,extension): 
                count += 1
    return count % 2 == 1

# Checks if any of the lines in polyA and polyB intersect. Then checks if any
# points in polyA are inside of polyB and vice versa.
def POLYSECT(polyA,polyB):
    for i in range(len(polyA)):
        p1 = polyA[i]
        p2 = polyA[(i+1)%len(polyA)]
        for j in range(len(polyB)):
            p3 = polyB[j]
            p4 = polyB[(j+1)%len(polyB)]
            if INTERSECT(p1,p2,p3,p4): 
                return True
    for point in polyA:
        if INSIDE(point,polyB): return True
    for point in polyB:
        if INSIDE(point,polyA): return True
    return False

def TESTPOLYSECT():
    print("Testing Polygon Intersection:")
    pA = [(3,8),(8,9),(10,6),(6,3)]
    pB = [(5,10),(0,7),(1,5),(4,4),(3,7)]
    pC = [(5,10),(0,7),(1,5),(4,4),(1,7)]
    pD = [(5,5),(6,5),(7,7)]
    pE = [(2,7),(3,8.2),(10,10),(11,2),(5,4)]
    print(chr(10209)+" Test Case 1:","PASSED" if POLYSECT(pA,pB) else "FAILED")
    print(chr(10209)+" Test Case 2:","PASSED" if not POLYSECT(pA,pC) else "FAILED")
    print(chr(10209)+" Test Case 3:","PASSED" if POLYSECT(pA,pD) else "FAILED")
    print(chr(10209)+" Test Case 4:","PASSED" if POLYSECT(pA,pE) else "FAILED")
    print(chr(10209)+" Test Case 5:","PASSED" if POLYSECT(pB,pC) else "FAILED")
    print(chr(10209)+" Test Case 6:","PASSED" if not POLYSECT(pB,pD) else "FAILED")
    print(chr(10209)+" Test Case 7:","PASSED" if POLYSECT(pB,pE) else "FAILED")
    print(chr(10209)+" Test Case 8:","PASSED" if not POLYSECT(pC,pD) else "FAILED")
    print(chr(10209)+" Test Case 9:","PASSED" if not POLYSECT(pC,pE) else "FAILED")
    print(chr(10209)+" Test Case 10:","PASSED" if POLYSECT(pD,pE) else "FAILED")
    print()

if __name__ == "__main__": TESTPOLYSECT()
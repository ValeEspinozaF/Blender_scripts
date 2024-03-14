# A Python3 program to find if 2 given line segments intersect or not 
# This code is contributed by Ansh Riyal 

import numpy as np
import pandas as pd
  
# Given three collinear points p, q, r, the function checks if  
# point q lies on line segment 'pr'  
def onSegment(p, q, r): 
    if ( (q[0] <= max(p[0], r[0])) and (q[0] >= min(p[0], r[0])) and 
        (q[1] <= max(p[1], r[1])) and (q[1] >= min(p[1], r[1]))): 
        return True
    return False
  
def orientation(p, q, r): 
    # to find the orientation of an ordered triplet (p,q,r) 
    # function returns the following values: 
    # 0 : Collinear points 
    # 1 : Clockwise points 
    # 2 : Counterclockwise 
      
    # See https://www.geeksforgeeks.org/orientation-3-ordered-points/amp/  
    # for details of below formula.  
      
    val = (float(q[1] - p[1]) * (r[0] - q[0])) - (float(q[0] - p[0]) * (r[1] - q[1])) 
    if (val > 0): 
          
        # Clockwise orientation 
        return 1
    elif (val < 0): 
          
        # Counterclockwise orientation 
        return 2
    else: 
          
        # Collinear orientation 
        return 0
    

def orientation_slope(p1, p2, p3):
    # Calculate slopes
    slope1 = (p2[1] - p1[1]) * (p3[0] - p2[0])
    slope2 = (p3[1] - p2[1]) * (p2[0] - p1[0])

    # Check orientation
    if slope1 == slope2:
        return 0 # Collinear
    elif slope1 < slope2:
        return 1 # CounterClockWise
    else:
        return 2 # ClockWise
    
def orientation_slope_df(p1, p2, p3):
    # Calculate slopes
    slope1 = (p2['lat'] - p1['lat']) * (p3['lon'] - p2['lon'])
    slope2 = (p3['lat'] - p2['lat']) * (p2['lon'] - p1['lon'])
    # Check orientation
    if slope1 == slope2:
        return 0 # Collinear
    elif slope1 < slope2:
        return 1 # CounterClockWise
    else:
        return 2 # ClockWise
  
  
# Returns true if the line segment 'p1q1' and 'p2q2' intersect. 
def doIntersect(p1,q1,p2,q2): 

    # Find the 4 orientations required for  
    # the general and special cases 
    o1 = orientation(p1, q1, p2) 
    o2 = orientation(p1, q1, q2) 
    o3 = orientation(p2, q2, p1) 
    o4 = orientation(p2, q2, q1) 
  
    # General case 
    if ((o1 != o2) and (o3 != o4)): 
        return True
  
    # Special Cases 
  
    # p1 , q1 and p2 are collinear and p2 lies on segment p1q1 
    if ((o1 == 0) and onSegment(p1, p2, q1)): 
        return True
  
    # p1 , q1 and q2 are collinear and q2 lies on segment p1q1 
    if ((o2 == 0) and onSegment(p1, q2, q1)): 
        return True
  
    # p2 , q2 and p1 are collinear and p1 lies on segment p2q2 
    if ((o3 == 0) and onSegment(p2, p1, q2)): 
        return True
  
    # p2 , q2 and q1 are collinear and q1 lies on segment p2q2 
    if ((o4 == 0) and onSegment(p2, q1, q2)): 
        return True
  
    # If none of the cases 
    return False


def point_InPolygon(polygon, point):

    n = len(polygon)

    # Create a point for line segment from p to infinite (a big value)
    point2 = [10000, point[1] + 0.00000001]

    # Count intersections of the above line with sides of polygon
    count = 0 

    for i in range(len(polygon)):
        next_ = (i + 1) % n
        
        # Check if the line 'point-point2' intersects the segment 'polygon[i]-polygon[next]'
        if (doIntersect(polygon[i], polygon[next_], point, point2)):
    
            # If the point is collinear with the segment 'i-next',
            if (orientation(polygon[i], point, polygon[next_]) == 0):
            
                # then check and return true if it lies on the segment.
                return onSegment(polygon[i], point, polygon[next_]);
        
            count += 1
        

    # Return true if count is odd, false otherwise
    return (count % 2 == 1)


def points_InPolygon(polygon, points):
    
    return [point_InPolygon(polygon, point) for point in points]


def polysegments_inPolygon(polygon, polysegments):
    
    n = len(polygon)
    m = len(polysegments)
    
    for i in range(len(polygon)):
        next_i = (i + 1) % n
        
        for j in range(len(polysegments)):
            next_j = (j + 1) % m
        
            # Check if the polysegment intersects the polygon
            if (doIntersect(polygon[i], polygon[next_i], polysegments[j], polysegments[next_j])):
                return True

    return False
    
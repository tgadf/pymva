"""
The Ramer-Douglas-Peucker algorithm roughly ported from the pseudo-code provided
by http://en.wikipedia.org/wiki/Ramer-Douglas-Peucker_algorithm
"""

from numpy import ndarray, array_equal, asarray, arccos, sqrt, diff, where, pi
from pandas.core.frame import DataFrame
import matplotlib.pyplot as plt


class RamerDouglasPeucker():

    def __init__(self, points, epsilon = None, min_angle = None):
        
        if isinstance(points, ndarray):
            self.points  = points
        elif isinstance(points, DataFrame):
            self.points  = points.values
        else:
            raise ValueError("Unknown data type {0}".format(type(points)))
            
        if epsilon is None:
            self.epsilon = 10
        else:
            self.epsilon = epsilon
            
        if min_angle is None:
            self.min_angle = 15.0/180.0 * pi
        else:
            self.min_angle = min_angle
            
        self.path     = None
        self.theta    = None
        self.turnings = None

    
    def compute(self, points = None, epsilon = None):
        """
        Reduces a series of points to a simplified version that loses detail, but
        maintains the general shape of the series.
        """
        dmax = 0.0
        index = 0
        if epsilon is None:
            epsilon = self.epsilon
        if points is None:
            points = self.points
            
        start = points[0]
        end   = points[-1]
        
        for i in range(1, len(points) - 1):
            point = points[i]
            d = self.point_line_distance(point, start, end)
            if d > dmax:
                index = i
                dmax = d
        if dmax >= epsilon:
            results = self.compute(points[:index+1], epsilon)[:-1] + self.compute(points[index:], epsilon)
        else:
            results = [points[0], points[-1]]

        return results


    def setPath(self):
        results = self.compute()        
        results = asarray(results)
        self.path = results


    def getPath(self):
        if self.path is None:
            self.setPath()
        return self.path
    
    
    def setAngles(self):
        if self.path is None:
            self.setPath()

        ## Get simplified path from RDP algorithm            
        sx, sy = self.path.T
    
        # compute the direction vectors on the simplified curve
        self.theta = self.angles()
        
        
    def setTurningPoints(self, min_angle = None):
        if self.theta is None:
            self.setAngles()
            
        if min_angle is None:
            min_angle = self.min_angle

        # Select the index of the points with the greatest theta
        # Large theta is associated with greatest change in direction.
        idx = where(self.theta>min_angle)[0]+1
        self.turnings = self.path[idx]
    
    
    def getTurningPoints(self, min_angle = None):
        if self.turnings is None or min_angle is not None:
            self.setTurningPoints(min_angle)
        return self.turnings


    def point_line_distance(self, point, start, end):
        
        if array_equal(start, end) is True:
            return self.distance(point, start)
        else:
            n = abs(
                (end[0] - start[0]) * (start[1] - point[1]) - (start[0] - point[0]) * (end[1] - start[1])
            )
            d = sqrt(
                (end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2
            )
            return n / d


    def distance(a, b):
        return  sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)
    
    
    def angles(self):
        directions = diff(self.path, axis=0)
        dir2 = directions[1:]
        dir1 = directions[:-1]
        return arccos((dir1*dir2).sum(axis=1)/(
                sqrt((dir1**2).sum(axis=1)*(dir2**2).sum(axis=1))))


    def plot(self):        
        if self.path is None:
            self.setPath()
        if self.theta is None:
            self.setAngles()
        if self.turnings is None:
            self.setTurningPoints()
            
        x, y   = self.points.T
        sx, sy = self.path.T
        tpx, tpy = self.turnings.T
            
        fig = plt.figure()
        ax =fig.add_subplot(111)
        
        ax.plot(x, y, 'b-', label='original path')
        ax.plot(sx, sy, 'g--', label='simplified path')
        ax.plot(tpx, tpy, 'ro', markersize = 10, label='turning points')
        ax.invert_yaxis()
        plt.legend(loc='best')
        plt.show()
        
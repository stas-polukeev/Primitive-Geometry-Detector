resolution = 256

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import json

class Figure:
    '''
    
    Basic parent class for other shapes.

    '''

    def __init__(self, mask, color, id):
        self.mask = mask 
        self.id = id
        self.color = color

    def can_place(self,  is_free, mask=None):
        '''
        Check if the position is free.

        '''
        if mask is None:
            mask = self.mask
        return np.all(is_free[mask])
        
    def get_bbox(self):
        '''
        Give the bounding box parameters, given the mask of the figure.
        '''
        Y, X = np.where(self.mask)
        y, x, h, w = Y.min(), X.min(), Y.max() - Y.min(), X.max() - X.min()
        del Y, X
        return y, x, h, w

    def draw_bbox(self, fig, color = np.array([255, 255, 255])):
        """
        Paint bounding box over the object.
        
        """
        y, x, h, w = self.get_bbox()
        fig[y : y + h + 1, x] = color
        fig[y : y + h + 1, x + w] = color
        fig[y, x : x + w + 1] = color
        fig[y + h, x : x + w + 1] = color
        return fig
    
    def get_param_dict(self):
        """
        Get parameters dict to convert to json or to train in-place.
        """
        keys = ['id', 'name', 'y', 'x', 'h', 'w'] 
        param_list = [self.id, self.name, *[int(i) for i in self.get_bbox()]]
        params = {keys[i] : param_list[i] for i in range(len(keys))}
        return params

    def draw(self, figure, is_free):
        is_free[self.mask] = 0
        figure[self.mask, :] = self.color
        return figure, is_free



class Circle(Figure):
    '''
    
    Circle with center cooridnates given by vector center and radius r.

    '''
    def __init__(self, center=None, r=None, color=None, id=None, is_free=None, sample=True):
        self.name = 'Circle'
        self.id = id
        if sample:
            can_place = False
            while not can_place:
                self.center, self.r, self.color = self.sample_params(is_free)
                mask = self.get_mask()
                can_place = super().can_place(is_free, mask)
            super().__init__(mask, self.color, self.id)
        else:
            self.center = center
            self.r = r 
            self.color = color
            self.id = id
            mask = self.get_mask()
            super().__init__(mask, self.color, self.id)
        


    def sample_params(self, is_free, fail_count=0):
        """
        Sample the circle in the free area.

        """
        l = 0
        while l == 0:
            free = np.indices((resolution, resolution))[:, is_free].T
            self.r = np.random.randint(25, 75)
            free = free[np.all(free > self.r, axis=1) * np.all(free < 256 - self.r, axis=1)]
            l = len(free)
        self.center = free[np.random.randint(len(free))]
        self.color = np.random.randint(0, 256, 3) 

        return self.center, self.r, self.color #may not be able to be placed
        
    def get_mask(self):
        mask = np.zeros((resolution, resolution), dtype = bool)
        mask[np.sum((np.indices((resolution, resolution)) - self.center.reshape(2, 1, 1)) ** 2, axis=0) < self.r ** 2] = True
        return mask

    def in_borders(self):
        return (self.center[0] - self.r >= 0) and (self.center[1] - self.r >= 0) and (self.center[0] + self.r <= resolution) and (self.center[1] + self.r <= resolution)
    


class Rhombus(Figure):
    
    '''
    Rhombus given by parameters p, q, center and alpha, color.
    p and q are the halves of 
    the main diagonals, center is the postion of crossing of main diagonals and alpha(radians) is 
    the angle between the diagonal p and the horizontal axis.
    '''

    def __init__(self, p=None, q=None, center=None, alpha=None, color=None, id=None, is_free=None, sample=True):
        
        self.name = 'Rhombus'
        self.id = id
        self.points = []
        if sample:
            can_place = False
            while not can_place:
                self.points = []
                p, q, center, alpha = self.sample_params(is_free)
                vectors = [np.array([p * np.sin(-alpha) #vertexes vectors relative to the center
                            ,p * np.cos(-alpha)])
                            ,np.array([q * np.cos(-alpha)
                            ,-q * np.sin(-alpha)])]
                for i in range(4):
                    self.points.append(center + (-1) ** (i < 2) * vectors[(i + 1) % 2])
                mask = self.get_mask()
                can_place = super().can_place(is_free, mask)

            super().__init__(mask, np.random.randint(0, 256, 3), self.id)
        
        else:
            vectors = [np.array([p * np.sin(-alpha) #vertexes vectors relative to the center
                        ,p * np.cos(-alpha)])
                        ,np.array([q * np.cos(-alpha)
                        ,-q * np.sin(-alpha)])]
            for i in range(4):
                self.points.append(center + (-1) ** (i < 2) * vectors[(i + 1) % 2])
            mask = self.get_mask()

            super().__init__(mask, color, self.id)
        
    def sample_params(self, is_free):
        l = 0
        while l == 0:
            free = np.indices((resolution, resolution))[:, is_free].T
            p = np.random.uniform(12.5, 75)
            q = np.random.uniform(12.5, 75)
            alpha = np.random.uniform(0, np.pi/2)
            y = max(p * np.sin(alpha), q * np.cos(alpha))
            x = max(p * np.cos(alpha), q * np.sin(alpha))
            x_mask = (free[:, 1] > x) * (free[:,1] < 256 - x)
            y_mask = (free[:, 0] > y) * (free[:,0] < 256 - y)
            free = free[x_mask * y_mask]
            l = len(free)
        center = free[np.random.randint(0, len(free))]
        return p, q, center, alpha       

    def get_mask(self):
        cross = []
        for i in range(4):
            cross += [self.points[i].reshape(2, 1, 1) - np.stack(np.indices((resolution, resolution)))]
        inside_check = []
        for i in range(4):
            inside_check.append(np.cross(cross[i % 4], cross[(i + 1) % 4], axis=0) >= 0)
        mask = np.stack(inside_check).all(axis=0)
        return mask
        

    def in_borders(self):
        a = True
        for i in range(4):
            for j in range(2):
                a *= self.points[i][j] >= 0 and self.points[i][j] <= resolution
        return a
    


class Triangle(Figure):

    '''

    Triangle given by three points a, b, c, color and id.

    '''

    def __init__(self, a=None, b=None, c=None, color=None, id=None, is_free=None, sample=True):
        #give points the orientation
        self.name = 'Triangle'
        self.id = id
        if sample:
            can_place = False
            while not can_place:
                self.points = self.sample_params(is_free)
                mask = self.get_mask()
                can_place = super().can_place(is_free, mask)
            super().__init__(mask, np.random.randint(0, 256, 3), self.id)
        else:
            self.points = [a]
            self.name = Triangle
            if np.cross(b - a, c - a) >= 0:
                self.points += [b, c]
            else:
                self.points += [c, b]
            mask = self.get_mask()
            super().__init__(mask, color, self.id)

    def sample_params(self, is_free):
        l = 0
        while l == 0:
            free = np.indices((resolution, resolution))[:, is_free].T
            a1, a2 = np.random.uniform(25, 150, size=2)
            alpha = np.random.uniform(1 / 7, np.pi / 2)
            beta = np.random.uniform(-np.pi/2, np.pi / 2)
            corners = np.array([[a2 * np.sin(alpha + beta), a2 * np.cos(alpha + beta)],[a1 * np.sin(beta), a1 * np.cos(beta)]])
            h_min, h_max = corners[:, 0].min(), corners[:, 0].max()
            w_min, w_max = corners[:, 1].min(), corners[:, 1].max()
            if (not 24 < h_max - h_min < 151) or (not 24 < w_max - w_min < 151):
                l = 0
            else:
                free = free[(free[:, 0] >= -h_min) * (free[:, 0] <= 256 - h_max) * (free[:, 1] >= -w_min) * (free[:, 1] <= 256 - w_max)]
                l = len(free)
        center = free[np.random.randint(l)]
        a = center
        b = center + a2 * np.array([np.sin(alpha + beta)
                                    ,np.cos(alpha + beta)])
        c = center + a1 * np.array([np.sin(beta)
                                    ,np.cos(beta)])
        return [a, b, c]   
    
    def get_mask(self):
        mask = np.zeros((resolution, resolution))
        cross = []
        for i in range(3):
            cross += [self.points[i].reshape(2, 1, 1) - np.stack(np.indices((resolution, resolution)))]
        inside_check = []
        for i in range(3):
            inside_check.append(np.cross(cross[i % 3], cross[(i + 1) % 3], axis=0) >= 0)
        mask = np.stack(inside_check).all(axis=0)
        return mask
    
    def in_borders(self):
        a = True
        for point in [self.a, self.b, self.c]:
            a *= point.max() <= 255 and point.min() >= 0
        return a 


class Hexagon(Figure):
    '''
    
    Hexagon given by the half diameter r, center location center, and angle alpha of rotation

    '''

    def __init__(self,center=None, r=None, alpha=None, color=None, id=None, is_free=None, sample=True):

        self.name = 'Hexagon'
        self.id = id
        if sample:
            can_place = False
            while not can_place:
                center, r, alpha = self.sample_params(is_free)
                angles = alpha + np.linspace(0, 5/3 * np.pi, 6)
                self.vertex = center.reshape(1, 2) + r * np.array([-np.sin(angles)
                                                                ,np.cos(angles)]).T
                mask = self.get_mask()
                can_place = super().can_place(is_free, mask) * self.in_borders()
            super().__init__(mask, np.random.randint(0, 256, 3), self.id)
        else:
            angles = alpha + np.linspace(0, 5/3 * np.pi, 6)
            self.vertex = center.reshape(1, 2) + r * np.array([-np.sin(angles)
                                                            ,np.cos(angles)]).T
            mask = self.get_mask()
            super().__init__(mask, color, self.id)

    def sample_params(self, is_free):
        l = 0
        while l == 0:
            free = np.indices((resolution, resolution))[:, is_free].T
            r = np.random.uniform(12.5, 75)
            alpha = np.random.uniform(0, np.pi / 2)
            angles = alpha + np.linspace(0, 2 * np.pi / 3, 3)
            r_projections = r * np.stack([np.sin(angles),
                                        np.cos(angles)])
            x  = r_projections[:, 1].max()
            y  = r_projections[:, 0].max()
            x_mask = (free[:, 1] > x) * (free[:, 1] < 256 - x)
            y_mask = (free[:, 0] > y) * (free[:, 0] < 256 - y)
            free = free[x_mask * y_mask]
            l = len(free)
        center = free[np.random.randint(len(free))]
        
        return center, r, alpha
    
    def in_borders(self):
        a = True
        return a * (self.vertex.max() <= 255) * (self.vertex.min() >= 0)
    
    def get_mask(self):
        cross = []
        for i in range(6):
            cross += [self.vertex[i].reshape(2, 1, 1) - np.stack(np.indices((resolution, resolution)))]
        inside_check = []
        for i in range(6):
            inside_check.append(np.cross(cross[i % 6], cross[(i + 1) % 6], axis=0) >= 0)
        mask = np.stack(inside_check).all(axis=0)
        return mask
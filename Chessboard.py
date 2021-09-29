import matplotlib.pyplot as plt
import cv2
import numpy as np
from numpy.core.fromnumeric import reshape

class ChessBoard():
    def __init__(self, width, height, grid_size):
        self.width = width
        self.height = height
        self.grid_size = grid_size
        self.image = None
        self.drawImage = None
    
    def loadImg(self, path):
        self.image = cv2.imread(path)

    def getCorners(self, coord = None):
        retval, self.corners = cv2.findChessboardCorners(self.image ,(self.width, self.height))

        if coord == None:
            return self.corners
        else:
            corner_point_id = coord[1] * self.width + coord[0]
            corner_point = tuple(map(int, self.corners[corner_point_id][0]))
            return corner_point

    def getWorldCorners(self, coord = None):
        if coord == None:
            width_range = np.arange(0, self.width * self.grid_size, self.grid_size)
            height_range = np.arange(0, self.height * self.grid_size, self.grid_size)
            x, y = np.meshgrid(width_range, height_range)
            self.world_corners = np.stack((x, y), axis = 2).reshape((54, 2))
            return self.world_corners
        else:
            corner_point = (coord[0] * self.grid_size, coord[1] * self.grid_size)
            return corner_point
    
    def drawCorners(self, coord = None):
        self.drawImage = self.image
        if coord == None:
            for corner_point_id in range(self.width * self.height):
                self.drawImage = cv2.circle(self.drawImage, \
                    tuple(map(int, self.corners[corner_point_id][0])), 2, (0, 255, 0), 4)
        else:
            corner_point_id = coord[1] * self.width + coord[0]
            self.drawImage = cv2.circle(self.drawImage, \
                    tuple(map(int, self.corners[corner_point_id][0])), 2, (0, 255, 0), 4)

    def getImg(self):
        return self.image, self.drawImage

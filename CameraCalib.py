import matplotlib.pyplot as plt
import cv2
import numpy as np
from numpy.core.fromnumeric import reshape

import sys
sys.path.append("../")
from Chessboard import ChessBoard

class CameraCalib():
    def __init__(self, width, height, grid_size, image_range):
        self.width = width
        self.height = height
        self.grid_size = grid_size
        self.world_points = []
        self.image_points = []
        self.chessboards = []

        for i in image_range:
            chessboard = ChessBoard(9, 6, 30)
            chessboard.loadImg("./data/left" + str(i).zfill(2) + ".jpg")
            self.chessboards.append(chessboard)
        
        self.image_size = self.chessboards[0].getImg()[0].shape

    def Calibrate(self):
        for chessboard in self.chessboards:
            corners = chessboard.getCorners()
            self.image_points.append(corners)
            world_corners = np.array(chessboard.getWorldCorners())
            self.world_points.append(world_corners)

        self.image_points = np.array(self.image_points).squeeze().astype(np.float32)
        self.world_points = np.array(self.world_points)
        ones = np.zeros((self.world_points.shape[0], self.world_points.shape[1], 1))
        self.world_points = np.concatenate((self.world_points, ones), axis = 2).astype(np.float32)

        retval, self.intrinsics, self.distortion, rvecs, tvecs = \
                cv2.calibrateCamera(self.world_points, self.image_points, self.image_size[:2], None, None)

        return self.intrinsics, self.distortion

if __name__ == "__main__":
    camer_calib = CameraCalib(9, 6, 30, range(1, 15))
    intrinsics, distortion = camer_calib.Calibrate()

    print("intrinsics:\n", intrinsics)
    print("distortion:\n", distortion[0])
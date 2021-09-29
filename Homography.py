import matplotlib.pyplot as plt
import cv2
import numpy as np
from numpy.core.fromnumeric import reshape
import sys
sys.path.append("../")
from Chessboard import ChessBoard

class Homography(ChessBoard):
    def __init__(self, width, height, grid_size):
        super().__init__(width, height, grid_size)

    def getHomography(self):
        A = []
        b = []
        for coord_x in range(self.width):
            for coord_y in range(self.height):
                coord = (coord_x, coord_y)
                u1, v1 = self.getWorldCorners(coord)
                u2, v2 = self.getCorners(coord)
                A_line_x = [u1, v1, 1, 0, 0, 0, -u1 * u2, -v1 * u2]
                b_line_x = u2
                A.append(A_line_x)
                b.append(b_line_x)
                A_line_y = [0, 0, 0, u1, v1, 1, -u1 * v2, -v1 * v2]
                b_line_y = v2
                A.append(A_line_y)
                b.append(b_line_y)

        A = np.array(A)
        b = np.array(b)

        x = np.linalg.inv(A.transpose().dot(A)).dot(A.transpose()).dot(b)
        self.Homo = np.append(x,1).reshape((3, 3))

        return self.Homo


if __name__ == "__main__":
    homo_chessboard = Homography(9, 6, 30)       # grid_size : mm
    homo_chessboard.loadImg("./data/left7.jpg")
    
    corner_point_coord = (2, 5)             # (x, y)
    corner_point = homo_chessboard.getCorners(corner_point_coord)
    print("corner_point:\n", corner_point)
    world_corner_point = homo_chessboard.getWorldCorners(corner_point_coord)
    print("world_corner_point:\n", world_corner_point)

    homo_chessboard.drawCorners(corner_point_coord)
    image, drawImage = homo_chessboard.getImg()

    Homo = homo_chessboard.getHomography()
    print("Homography Mat:\n", Homo)

    world_corner_point = np.array(world_corner_point)
    world_corner_point = np.append(world_corner_point, 1)
    Homo_coord = Homo.dot(world_corner_point)

    print("Translated corner point:\n", tuple(Homo_coord[:2]/Homo_coord[2]))

    plt.figure()
    plt.imshow(drawImage)
    plt.show()

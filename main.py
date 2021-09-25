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

    def getCorners(self, coord):
        retval, self.corners = cv2.findChessboardCorners(self.image ,(9, 6))
        corner_point_id = coord[1] * self.width + coord[0]
        corner_point = tuple(map(int, self.corners[corner_point_id][0]))

        return corner_point

    def getWorldCorners(self, coord):

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

    def getHomography(self):
        A = []
        b = []
        for coord_x in range(self.width):
            for coord_y in range(self.height):
                coord = (coord_x, coord_y)
                u1, v1 = self.getCorners(coord)
                u2, v2 = self.getWorldCorners(coord)
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

        print(A)
        print(b)

        x = np.linalg.inv(A.transpose().dot(A)).dot(A.transpose()).dot(b)

        self.Homo = np.append(x,1).reshape((3, 3))

        return self.Homo


if __name__ == "__main__":
    chessboard = ChessBoard(9, 6, 30)       # grid_size : mm
    chessboard.loadImg("./data/left7.jpg")
    
    corner_point_coord = (2, 5)             # (x, y)
    corner_point = chessboard.getCorners(corner_point_coord)
    print("corner_point:\n", corner_point)
    world_corner_point = chessboard.getWorldCorners(corner_point_coord)
    print("world_corner_point:\n", world_corner_point)

    chessboard.drawCorners(corner_point_coord)
    image, drawImage = chessboard.getImg()

    Homo = chessboard.getHomography()

    print("Homography Mat:\n", Homo)

    corner_point = np.array(corner_point)
    corner_point = np.append(corner_point, 1)
    Homo_coord = Homo.dot(corner_point)

    print(world_corner_point)
    print(Homo_coord[:2]/Homo_coord[2])

    plt.figure()
    plt.imshow(drawImage)
    plt.show()

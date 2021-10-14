import numpy as np
import cv2
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class PhotometricStereo():
    def __init__(self):
        self.images = []
        self.light_coords = []
        self.image_num = 0

    def loadImgNCoord(self, paths, light_coords):
        for path in paths:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            self.images.append(img)
            self.image_num = self.image_num + 1
    
        self.light_coords = light_coords
        self.img_size = self.images[0].shape

    def calNormal(self):
        self.normal_map = np.zeros((self.img_size[0], self.img_size[1], 3))
        self.rou_map = np.zeros(self.img_size)
        
        light_coords = np.array(self.light_coords)
        for i in range(self.img_size[0]):
            for j in range(self.img_size[1]):
                intensities = []
                for img_num in range(self.image_num):
                    intensities.append(self.images[img_num][i][j])
                
                intensities = np.array(intensities)
                normal = np.linalg.inv(light_coords.transpose().dot(light_coords))\
                                        .dot(light_coords.transpose()).dot(intensities)
                rou = np.linalg.norm(normal)
                normal = normal / rou
                self.rou_map[i][j] = rou
                self.normal_map[i][j] = normal

    def surfaceReconstruct(self):
        self.z_map = np.zeros(self.img_size)
        for i in range(self.img_size[0]):
            for j in range(self.img_size[1]):
                di = -self.normal_map[i][j][0] / self.normal_map[i][j][2]
                dj = -self.normal_map[i][j][1] / self.normal_map[i][j][2]
                
                if i - 1 >= 0:
                    self.z_map[i][j] = self.z_map[i - 1][j] + di
                elif j - 1 >= 0:
                    self.z_map[i][j] = self.z_map[i][j - 1] + dj
                else:
                    self.z_map[i][j] = 0
    
    def getZ(self):

        return self.z_map

if __name__ == "__main__":

    photometricStereo = PhotometricStereo()
    paths = ["./data/im1.png", "./data/im2.png", "./data/im3.png", "./data/im4.png"]
    light_coords = [(0, 0, -1), (0, 0.2, -1), (0, -0.2, -1), (0.2, 0, -1)]
    photometricStereo.loadImgNCoord(paths, light_coords)
    photometricStereo.calNormal()
    photometricStereo.surfaceReconstruct()
    
    fig=plt.figure()
    ax = Axes3D(fig)

    xx = np.arange(0, 100)
    yy = np.arange(0, 100)
    X, Y = np.meshgrid(xx, yy)
    Z = photometricStereo.getZ()
    ax.plot_surface(X,Y,Z,cmap='rainbow')
    
    plt.show()
    
import matplotlib.pyplot as plt
import cv2
import numpy as np

class SkinDet():

    def __init__(self):
        pass

    def loadImg(self, path):
        self.image = cv2.imread(path)

    def preProcess(self):

        self.imageYCbCr = cv2.cvtColor(self.image, cv2.COLOR_BGR2YCR_CB)
        self.imageCr = self.imageYCbCr[:, :, 2]
        return self.imageCr

    def detSkin(self):
        self.skinMask = np.zeros_like(self.imageCr)
        for row in range(self.imageYCbCr.shape[0]):
            for col in range(self.imageYCbCr.shape[1]):
                Cr = self.imageCr[row][col]
                if Cr <= 120 and Cr >= 85:
                    self.skinMask[row][col] = 1
        return self.skinMask

if __name__ == "__main__":
    skindet = SkinDet()

    skindet.loadImg("./data/skin0.jpg")
    cr = skindet.preProcess()
    skin = skindet.detSkin()


    plt.figure()
    plt.imshow(cr)
    plt.figure()
    plt.imshow(skin)
    plt.show()

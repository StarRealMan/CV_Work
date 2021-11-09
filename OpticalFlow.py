import matplotlib.pyplot as plt
import cv2
import numpy as np

class OpticalFlow():

    def __init__(self):
        self.imgseq = []
        self.opticalflowseq = []
        self.gradientseq = []

    def loadImgSeq(self, path, seq_name, seq_size):
        for img_num in range(seq_size):
            raw_img = cv2.imread(path + seq_name + str(img_num) + ".pgm")
            gray_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
            self.imgseq.append(gray_img)
            self.opticalflowseq.append(np.zeros((gray_img.shape[0], gray_img.shape[1], 2)))
            sobelx = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize = 3)
            sobely = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize = 3)
            gradient = np.dstack((sobelx, sobely))
            self.gradientseq.append(gradient)

    def findNeighbor(self, u, v, size):
        umin = u - size
        umax = u + size
        vmin = v - size
        vmax = v + size

        img_u_max = self.imgseq[0].shape[1]
        img_v_max = self.imgseq[0].shape[0]

        if umin < 0:
            umin = 0
        if vmin < 0:
            vmin = 0
        if umax > img_u_max:
            umax = img_u_max
        if vmax > img_v_max:
            vmax = img_v_max

        return umin, umax + 1, vmin, vmax + 1

    def calLeastSquare(self, delta_intensity_patch, gradient_patch):
        A = []
        b = []
        for v in range(delta_intensity_patch.shape[0]):
            for u in range(delta_intensity_patch.shape[1]):
                gradient = gradient_patch[v][u]
                A_row = [gradient[0], gradient[1]]
                A.append(A_row)
                delta_intensity = delta_intensity_patch[v][u]
                b.append(-delta_intensity)

        A = np.array(A)
        b = np.array(b)

        if np.linalg.det(A.transpose().dot(A)) != 0:
            opticalflow = np.linalg.inv(A.transpose().dot(A)).dot(A.transpose()).dot(b)
        else:
            opticalflow = np.zeros((2))
        
        return opticalflow

    def calOpticalFlow(self, size):
        for v in range(self.imgseq[0].shape[0]):
            for u in range(self.imgseq[0].shape[1]):
                umin, umax, vmin, vmax = self.findNeighbor(u, v, 2)
                for img_num in range(len(self.imgseq) - 1):
                    img = self.imgseq[img_num]
                    gradient = self.gradientseq[img_num]
                    next_img = self.imgseq[img_num + 1]
                    next_gradient = self.gradientseq[img_num + 1]

                    patch = img[vmin:vmax, umin:umax]
                    next_patch = next_img[vmin:vmax, umin:umax]
                    gradient_patch = gradient[vmin:vmax, umin:umax]

                    opticalflow = self.calLeastSquare(next_patch - patch, gradient_patch)
                    
                    self.opticalflowseq[img_num][v][u] = opticalflow

        return self.opticalflowseq

if __name__ == "__main__":

    optical_flow = OpticalFlow()

    optical_flow.loadImgSeq("./data/sphere/", "sphere.", 2)
    ofseq = optical_flow.calOpticalFlow(5)

    plt.figure()
    plt.imshow(ofseq[0][:, :, 1])
    plt.show()
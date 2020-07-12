import numpy as np
import cv2
import random
from skimage.measure import compare_ssim as SSIM
from sklearn.metrics import mean_absolute_error as MAE
import time


def PSNR(original, compressed):
    from math import log10, sqrt
    mse = np.mean((original - compressed) ** 2)
    if (mse == 0):  # MSE is zero means no noise is present in the signal .
        # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr


class NoiseRemoval():
    def __init__(self, path=None):
        self.imagePath = path
        self.size = 3
        self.padding = 3
        self.window = 5
        # s & b vals as in paper
        self.s = 20
        self.b = 70
        if self.imagePath:
            self.openImage(self.imagePath)

    # Add salt and pepper noise where prob = noise %
    def add_noise(self, image, prob):
        prob /= 4
        output = np.zeros(image.shape, np.uint8)
        thres = 1 - prob
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                rdn = random.random()
                if rdn < prob:
                    output[i][j] = 0
                elif rdn > thres:
                    output[i][j] = 255
                else:
                    output[i][j] = image[i][j]
        return output

    # BIG function as in paper
    def __BIG(self, u):
        if u < self.s:
            return 0
        if u >= self.b:
            return 1
        if self.s <= u < self.b:
            return (u - self.s) / (self.b - self.s)

    # SMALL function as in paper
    def __SMALL(self, u):
        if u < self.s:
            return 1
        if u >= self.b:
            return 0
        if self.s <= u < self.b:
            return (u - self.b) / (self.s - self.b)

    def __Rules(self, Dl):
        Fs = [self.__SMALL(q) for q in Dl]
        Fb = [self.__BIG(q) for q in Dl]

        # eq 7(a)
        F1 = Fb[0] * Fb[1] * Fb[2] * Fb[3]
        # eq 7(b)
        F2 = Fs[0] * Fb[1] * Fb[2] * Fb[3]
        # eq 7(c)
        F3 = Fs[0] * Fs[1] * Fb[2] * Fb[3]
        # eq 7(d)
        F4 = Fs[0] * Fs[1] * Fs[2] * Fb[3]
        # eq 7(e)
        F5 = Fs[0] * Fs[1] * Fs[2] * Fs[3]

        # return the code of maximum Rule
        if F1 > max(F2, F3, F4, F5):
            return 1
        if F2 > max(F1, F3, F4, F5):
            return 2
        if F5 > max(F1, F2, F3, F4):
            return 5
        else:
            return 3

    # Loops through complete Image
    def fuzzyMedianFilter(self, img):

        result = np.copy(img)

        pad = int(self.window / 2)
        for i in range(pad, len(img) - pad):
            for j in range(pad, len(img[0]) - pad):
                # Select Window
                win = img[i - pad:i + pad, j - pad:j + pad]
                # Select Sk as in paper
                Sk = [
                    [img[i - 2, j - 2], img[i - 1, j - 1], img[i + 1, j + 1], img[i + 2, j + 2]],
                    [img[i, j - 2], img[i, j - 1], img[i, j + 1], img[i, j + 2]],
                    [img[i + 2, j - 2], img[i + 1, j - 1], img[i - 1, j + 1], img[i - 2, j + 2]],
                    [img[i - 2, j], img[i - 1, j], img[i + 1, j], img[i + 2, j]]
                ]
                Sk = np.array(Sk)
                # Current Pixel
                xij = img[i, j]

                result[i, j] = self.pixelFuzzyMedianFilter(Sk, xij, win)
        self.smooth_image = result
        return self.smooth_image

    # Works on a single window
    def pixelFuzzyMedianFilter(self, Sk, xij, win):

        # Dk as in paper
        Dk = np.sum(abs(Sk - xij), axis=1) / 4

        # sorted indexes of Dk
        Tk = np.argsort(Dk)

        # Dl as in paper
        Dl = np.sort(Dk)

        # Select rules based on Dl values
        rule = self.__Rules(Dl)

        # If F1 is max he current pixel is possibly a noise in the smooth region.
        # y[i;j] = MED(x).
        if rule == 1:
            return np.median(win)

        # If F2 is max the current pixel is an edge pixel and is preserved.
        # If F5 is max the current pixel is a noise-free pixel in a smooth region.
        if rule == 2 or rule == 5:
            return xij

        # If max is F3 or F4, the current pixel is possibly a noise in detailed region.
        if rule == 3:  # or 4
            if abs(Dl[0] - Dl[1]) >= abs(Dl[2] - Dl[3]):
                med = [x for x in Sk[Tk[0]]]
                med.append(xij)
                return np.median(med)
            else:
                med = [x for x in Sk[Tk[3]]]
                med.append(xij)
                return np.median(med)

    def imshow(self, img, win_name='win', delay=2):
        '''
        img : image to display
        win_name : Name of the window
        delay : display time(seconds)
        '''
        cv2.imshow(win_name, img)
        cv2.waitKey(delay=300)
        time.sleep(delay)

        cv2.destroyAllWindows()

    def openImage(self, path):
        self.imagePath = path
        #         org Image
        self.org_image = cv2.imread(self.imagePath, 0)
        self.org_image = cv2.resize(self.org_image, (600, 600), interpolation=cv2.INTER_AREA)
        self.image = self.org_image.copy()

    def __pad_with(self, vector, pad_width, iaxis, kwargs):
        pad_value = kwargs.get('padder', 0)
        vector[:pad_width[0]] = pad_value
        vector[-pad_width[1]:] = pad_value

    # Bubble sort to sort the elements for V0 to take median
    def __sort(self, w, n):
        V0 = []
        for x in range(n):
            for y in range(n):
                V0.append(w[x][y])

        count = n * n
        for i in range(count):
            for j in range(count - i - 1):
                if V0[j] > V0[j + 1]:
                    V0[j], V0[j + 1] = V0[j + 1], V0[j]

        return V0

    def __corrupted(self, img, X, Y, n):
        image = img

        i = int(n / 2)
        window = [image[x][Y - i: Y + i + 1] for x in range(X - i, X + i + 1)]
        V0 = self.__sort(window, n)

        mid = int((n * n) / 2)
        median = V0[mid]

        # The difference array
        Dv = []
        for x in range(1, len(V0)):
            Dv.append(V0[x] - V0[x - 1])

        # Finding maximum on left and right cluster of the median for vl and vr
        maximum = 0
        index1 = 0
        for x in range(0, mid):
            if Dv[x] > maximum:
                maximum = Dv[x]
                index1 = x
        vl = V0[index1]

        maximum = 0
        index2 = 0
        for x in range(mid, len(Dv)):
            if Dv[x] > maximum:
                maximum = Dv[x]
                index2 = x
        vr = V0[index2]

        # The three clusters formed by vl and vr
        cluster1 = V0[0:index1 + 1]
        cluster2 = V0[index1 + 1:index2 + 1]
        cluster3 = V0[index2 + 1:len(V0) + 1]

        if n != 3:
            # if n == size
            if image[X][Y] in cluster1:
                return 1  # uncorrupted
            elif image[X][Y] in cluster2:
                return 1  # uncorrupted

            else:
                return self.__corrupted(X, Y, 3)
        else:
            # if n == 3
            if image[X][Y] in cluster3:
                return 0  # corrupted
            else:
                return 1  # uncorrupted

    def findNoise(self, img, verbose=False):
        size = self.size
        halfSize = int(size / 2)
        C = np.copy(img)

        for x in range(halfSize, len(self.image) - halfSize):
            for y in range(halfSize, len(self.image[0]) - halfSize):
                C[x - halfSize, y - halfSize] = self.__corrupted(img, x, y, size)
        self.Corrupted_image = C * 255
        if verbose:
            print("\nMask image for corrupted pixels (C): \n", C)


# if __name__ == "__main__":

#     nr = NoiseRemoval('clean1.png')

#     nr.imshow(nr.image, 'Original-Image')

#     nr.image = nr.add_noise(nr.image, 0.40)

#     main_noise_image = np.copy(nr.image)

#     nr.imshow(nr.image, '40%-Noise-Image')

#     # nr.image is noise image

#     nr.findNoise(nr.image)

#     nr.imshow(nr.Corrupted_image, 'Corruped-Pixels-Image', delay=5)

#     p_smoothImg = np.copy(nr.image)

#     for i in range(3):
#         print(f'iteration# {i + 1}')

#         n_smoothImg = nr.fuzzyMedianFilter(p_smoothImg)

#         nr.imshow(n_smoothImg, f'ite:{i + 1}-Image', delay=5)

#         # nr.smooth_image is smooth image after each iteration

#         psnr = PSNR(n_smoothImg, p_smoothImg)
#         ssim, _ = SSIM(n_smoothImg, p_smoothImg, full=True)
#         mae = MAE(n_smoothImg, p_smoothImg)

#         print(f'PSNR Value for iteration{i + 1} : {psnr}')
#         print(f'SSIM Value for iteration{i + 1} : {ssim}')
#         print(f'MAE Value for iteration{i + 1} : {mae}')

#         p_smoothImg = n_smoothImg

#     abs_diff_image = np.abs(main_noise_image - n_smoothImg)

#     nr.imshow(abs_diff_image, 'Difference_image', delay=5)

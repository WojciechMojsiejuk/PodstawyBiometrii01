import cv2 as cv
import numpy as np

class Histogram:
    """logic dealing with histogram"""
    @staticmethod
    def compute_histogram(img, channel):
        image = np.array(img)
        return cv.calcHist(image, channel, None, [256],[0, 256])
    
    @staticmethod
    def equalize_histogram_grayscale(img):
        image = np.array(img)
        image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
        image  = cv.equalizeHist(image)
        image = cv.cvtColor(image, cv.COLOR_GRAY2RGB)
        return image
    
    @staticmethod    
    def equalize_histogram_YCrCb(img):
        image = np.array(img)
        y, cr, cb = cv.split(cv.cvtColor(image, cv.COLOR_RGB2YCrCb))
        y = cv.equalizeHist(y)
        image = cv.merge((y, cr, cb))
        image = cv.cvtColor(image, cv.COLOR_YCrCb2RGB)
        return image

    @staticmethod    
    def normalize_histogram(img, a, b):
        image = np.array(img)
        return cv.normalize(image, None, a, b, cv.NORM_MINMAX)

class Brightness:
    @staticmethod
    def gamma_correction(img, gamma):
        image = np.array(img)
        lookUpTable = np.empty((1,256), np.uint8)
        for i in range(256):
            lookUpTable[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
        return cv.LUT(image, lookUpTable)

class Conversion:
    @staticmethod
    def convert_2_gray(img):
        image = np.array(img)
        return cv.cvtColor(image, cv.COLOR_RGB2GRAY)

class GrayscaleConversionError(Exception):
    pass

class Binarization:
    @staticmethod
    def binary_thresholding(img, thresh):
        image = np.array(img)
        if len(image.shape) != 2:
            raise GrayscaleConversionError('Image needs to be converted to grayscale')
        _, result = cv.threshold(img,thresh,255,cv.THRESH_BINARY)
        return result

    @staticmethod
    def otsu(img):
        image = np.array(img)
        if len(image.shape) != 2:
            raise GrayscaleConversionError('Image needs to be converted to grayscale')
        _, result = cv.threshold(image,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
        return result

    @staticmethod
    def niblack(img, kernel, k):
        k = -k
        image = np.array(img)
        if len(image.shape) != 2:
            raise GrayscaleConversionError('Image needs to be converted to grayscale')
        image = cv.bitwise_not(image)
        return cv.ximgproc.niBlackThreshold(image, 255, cv.THRESH_BINARY_INV , kernel, k)


class Filter:

    filters = np.array(
        [
        [
            [0, 0, 0],
            [0, 0, 0],  #default
            [0, 0, 0]
        ],

        [
            [-1, 0, 1],
            [-1, 0, 1],  #Prewitt 0 degrees
            [-1, 0, 1]
        ],

        [
            [0, 1, 1],
            [-1, 0, 1],  #Prewitt 45 degrees
            [-1, -1, 0]
        ],
        [
            [1, 1, 1],
            [0, 0, 0],  #Prewitt 90 degrees
            [-1, -1, -1]
        ],
        [
            [1, 1, 0],
            [1, 0, -1],  #Prewitt 135 degrees
            [0, -1, -1]
        ],

        [
            [1, 0, -1],
            [1, 0, -1],  #Prewitt 180 degrees
            [1, 0, -1]
        ],

        [
            [0, -1, -1],
            [1, 0, -1],  #Prewitt 225 degrees
            [1, 1, 0]
        ],
        [
            [-1, -1, -1],
            [0, 0, 0],  #Prewitt 270 degrees
            [1, 1, 1]
        ],
        [
            [-1, -1, 0],
            [-1, 0, 1],  #Prewitt 315 degrees
            [0, 1, 1]
        ],

        [
            [-1, 0, 1],
            [-2, 0, 2],  #Sobel 0 degrees
            [-1, 0, 1]
        ],

        [
            [0, 1, 2],
            [-1, 0, 1],  #Sobel 45 degrees
            [-2, -1, 0]
        ],
        [
            [1, 2, 1],
            [0, 0, 0],  #Sobel 90 degrees
            [-1, -2, -1]
        ],
        [
            [2, 1, 0],
            [1, 0, -1],  #Sobel 135 degrees
            [0, -1, -2]
        ],

        [
            [1, 0, -1],
            [2, 0, -2],  #Sobel 180 degrees
            [1, 0, -1]
        ],

        [
            [0, -1, -2],
            [1, 0, -1],  #Sobel 225 degrees
            [2, 1, 0]
        ],
        [
            [-1, -2, -1],
            [0, 0, 0],  #Sobel 270 degrees
            [1, 2, 1]
        ],
        [
            [-2, -1, 0],
            [-1, 0, 1],  #Sobel 315 degrees
            [0, 1, 2]
        ],

        [
            [0, -1, 0],
            [-1, 4, -1],  #Laplace 1
            [0, -1, 0]
        ],
        [
            [-1, -1, -1],
            [-1, 8, -1],  #Laplace 2
            [-1, -1, -1]
        ],
        [
            [1, -2, 1],
            [-2, 4, -2],  #Laplace 3
            [1, -2, 1]
        ],

        [
            [1, 1, 1],
            [1, -2, -1],  #Edge Detection 1
            [1, -1, -1]
        ],

        [
            [1, 1, 1],
            [-1, -2, 1],  #Edge Detection 2
            [-1, -1, 1]
        ],
        [
            [1, -1, -1],
            [1, -2, -1],  #Edge Detection 3
            [1, 1, 1]
        ],
        [
            [-1, -1, 1],
            [-1, -2, 1],  #Edge Detection 4
            [1, 1, 1]
        ],

        ])

    @staticmethod
    def median(img, kernel):
          image = np.array(img)
          return cv.medianBlur(image, kernel)

    @staticmethod
    def linear_filter(img, kernel):
        image = np.array(img)
        return cv.filter2D(image,-1,kernel)

    @staticmethod
    def box_blur(img):
        image = np.array(img)
        kernel = np.ones((3,3),np.float32)/9
        return cv.filter2D(image,-1,kernel)
        
    @staticmethod
    def gaussian_blur(img):
        image = np.array(img)
        return cv.GaussianBlur(image,(5,5),0)

    @staticmethod
    def kuwahara_filter(img, kernel):
        image = np.array(img)
        shift = int((kernel + 1) / 2)
        height =  image.shape[0] - kernel
        width = image.shape[1] - kernel
        channels = image.shape[2]
        for y in range(height):
            for x in range(width):
                #devide into 4 regions
                region_NW = image[y:y+shift,x:x+shift,:]
                region_NE = image[y:y+shift,x+shift-1:x+kernel,:]
                region_SW = image[y+shift-1:y+kernel,x:x+shift,:]
                region_SE = image[y+shift-1:y+kernel,x+shift-1:x+kernel,:]
                
                #calculate avg and variance
                avg = []
                var = []
                avg.append(np.average(np.average(region_NW,1),0))
                var.append(np.var(region_NW))
                avg.append(np.average(np.average(region_NE,1),0))
                var.append(np.var(region_NE))
                avg.append(np.average(np.average(region_SW,1),0))
                var.append(np.var(region_SW))
                avg.append(np.average(np.average(region_SE,1),0))
                var.append(np.var(region_SE))
               

                #find minimum variance
                min_var_reg = int(np.argmin(np.array(var)))
                #print(min_var_reg)
                
                #result is avg of min var region
                for channel in range(channels):
                    image[y+shift-1,x+shift-1,channel] = avg[min_var_reg][channel]
        return image



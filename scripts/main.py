import cv2
import numpy as np
import utils

image = cv2.imread('maxim2.jpg')
haar_face_cascade = cv2.CascadeClassifier('../data/haarcascade_frontalface_alt.xml') # create cascade classifier for face detection

# Face finder

cropImage = utils.cut_face(haar_face_cascade, image, scaleFactor = 1.2)

# Blur image

blurredImage = cv2.GaussianBlur(cropImage, (5, 5), 0) # Remove noise

# Try grapCut

mask = np.zeros(blurredImage.shape[:2], np.uint8)

bgdModel = np.zeros((1,65), np.float64)

fgdModel = np.zeros((1,65), np.float64)

width, height = blurredImage.shape[:2]
rect = (0, 0, width - 1, height - 1)

cv2.grabCut(blurredImage, mask, rect, bgdModel, fgdModel, 9, cv2.GC_INIT_WITH_RECT)

mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')

img = blurredImage * mask2[:,:,np.newaxis]

cv2.imwrite('grapCut.jpg', img)

# # RGB -> Gray Scale

# grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# # Thresholding

# ret, imgf = cv2.threshold(grayImage, 177, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# cv2.imwrite('thresholding.jpg', imgf)

# # Find edges

# edgeImg = np.max( np.array([ utils.edgedetect(imgf[:,:, 0]), utils.edgedetect(imgf[:,:, 1]), utils.edgedetect(imgf[:,:, 2]) ]), axis=0 )

# mean = np.mean(edgeImg)
# # Zero any value that is less than mean. This reduces a lot of noise.
# edgeImg[edgeImg <= mean] = 0

# cv2.imwrite('edged-image.jpg', edgeImg)


# edgeImg_8u = np.asarray(edgeImg, np.uint8)
# # Find contours
# significant = utils.findSignificantContours(cropImage, edgeImg_8u)

# # Mask
# mask = edgeImg.copy()
# mask[mask > 0] = 0
# cv2.fillPoly(mask, significant, 255)
# # Invert mask
# mask = np.logical_not(mask)

# #Finally remove the background
# cropImage[mask] = 0

# cv2.imwrite('without-background.jpg', cropImage)
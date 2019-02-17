import cv2
import numpy as np

# im = cv2.imread('maxim.jpg')
# height, width, depth = im.shape

# circle_mask = np.zeros((height, width), np.uint8)

# cv2.circle(circle_mask, (round(width/2), round(height/2)), 150, 1, thickness = -1)

# cv2.imwrite("masked2.png", circle_mask)

# bgdModel = np.zeros((1,65),np.float64)
# fgdModel = np.zeros((1,65),np.float64)

# mask = np.zeros(circle_mask.shape[:2],np.uint8)
# mask[circle_mask == 0] = 0
# mask[circle_mask == 255] = 1

# mask, bgdModel, fgdModel = cv2.grabCut(im, mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
# mask = np.where((mask==2)|(mask==0),0,1).astype('uint8')
# img = im*mask[:,:,np.newaxis]

# cv2.imwrite('masked2.jpg', img)

# masked_data = cv2.bitwise_and(im, im, mask=circle_mask)

# cv2.imwrite("masked.jpg", masked_data)


img = cv2.imread('maxim.jpg')

height, width, depth = img.shape
circle_mask = np.zeros((height, width), np.uint8) # mask init
cv2.circle(circle_mask, (round(width / 2), round(height / 2)), 100, 1, thickness = -1) # black mask with white circle
# circle_mask = cv2.bitwise_not(circle_mask) # white mask with black circle

bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)

width, height = img.shape[:2]
rect = (300, 0, width, height)

cv2.grabCut(img, circle_mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

mask2 = np.where((circle_mask==2)|(circle_mask==0), 0, 1).astype('uint8')

image = img * mask2[:,:,np.newaxis]

cv2.imwrite('masked.jpg', image)

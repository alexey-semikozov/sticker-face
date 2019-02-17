import cv2
import numpy as np
from PIL import Image

img_input = np.array(Image.open('IMG_6407.jpg'))
start_input = cv2.resize(img_input, dsize=(300, 150), interpolation=cv2.INTER_CUBIC)
cv2.imwrite('maxim-min4.jpg', start_input)

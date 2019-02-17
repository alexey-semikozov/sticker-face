import cv2
import numpy as np
from PIL import Image
import queue
import itertools as it #from more_itertools import chunked

def detect_faces(f_cascade, colored_img, scaleFactor = 1.1):
    #just making a copy of image passed, so that passed image is not changed
    img_copy = colored_img.copy()

    #convert the test image to gray image as opencv face detector expects gray images
    gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)

    #let's detect multiscale (some images may be closer to camera than others) images
    faces = f_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=5)

    #go over list of faces and draw them as rectangles on original colored img
    for (x, y, w, h) in faces:
        cv2.rectangle(img_copy, (x, y), (x+w, y+h), (0, 255, 0), 2)

    return img_copy

# for one face on the image
def cut_face(f_cascade, colored_img, scaleFactor = 1.1):
    #just making a copy of image passed, so that passed image is not changed 
    img_copy = colored_img.copy()

    #convert the test image to gray image as opencv face detector expects gray images
    gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)

    #let's detect multiscale (some images may be closer to camera than others) images
    faces = f_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=5)

    #go over list of faces and draw them as rectangles on original colored img
    i = 0
    for (x, y, w, h) in faces:
        r = max(w, h) / 2
        centerx = x + w / 2
        centery = y + h / 2
        nx = int(centerx - r)
        ny = int(centery - r)
        nr = int(r * 2)

        faceimg = img_copy[ny:ny+nr, nx:nx+nr]
        i += 1
        # cv2.imwrite("image%d.jpg" % i, faceimg)

    return faceimg

def edgedetect (channel):
    sobelX = cv2.Sobel(channel, cv2.CV_16S, 1, 0)
    sobelY = cv2.Sobel(channel, cv2.CV_16S, 0, 1)
    sobel = np.hypot(sobelX, sobelY)

    sobel[sobel > 255] = 255; # Some values seem to go above 255. However RGB channels has to be within 0-255
    return sobel

def findSignificantContours (img, edgeImg):
    contours, heirarchy = cv2.findContours(edgeImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Find level 1 contours
    level1 = []
    for i, tupl in enumerate(heirarchy[0]):
        # Each array is in format (Next, Prev, First child, Parent)
        # Filter the ones without parent
        if tupl[3] == -1:
            tupl = np.insert(tupl, 0, [i])
            level1.append(tupl)

    # From among them, find the contours with large surface area.
    significant = []
    tooSmall = edgeImg.size * 5 / 100 # If contour isn't covering 5% of total area of image then it probably is too small
    for tupl in level1:
        contour = contours[tupl[0]];
        area = cv2.contourArea(contour)
        if area > tooSmall:
            significant.append([contour, area])

            # Draw the contour on the original image
            cv2.drawContours(img, [contour], 0, (0,255,0),2, cv2.LINE_AA, maxLevel=1)

    significant.sort(key=lambda x: x[1])
    #print ([x[1] for x in significant]);
    return [x[0] for x in significant]

def scale_image(start_image, width = None, height = None):
    w, h = start_image.size
 
    if width and height:
        max_size = (width, height)
    elif width:
        max_size = (width, h)
    elif height:
        max_size = (w, height)
    else:
        # No width or height specified
        raise RuntimeError('Width or height required!')
 
    start_image.thumbnail(max_size, Image.ANTIALIAS)
    return start_image

# from habr blog post
def dijkstra(start_points, w):
    d = np.zeros(w.shape) + np.infty
    v = np.zeros(w.shape, dtype=np.bool)
    q = queue.PriorityQueue()
    for x, y in start_points:
        d[x, y] = 0
        q.put((d[x, y], (x, y)))

    for x, y in it.product(range(w.shape[0]), range(w.shape[1])):
        if np.isinf(d[x, y]):
            q.put((d[x, y], (x, y)))

    while not q.empty():
        _, p = q.get()
        if v[p]:
            continue

        neighbourhood = []
        if p[0] - 1 >= 0:
            neighbourhood.append((p[0] - 1, p[1]))
        if p[0] + 1 <= w.shape[0] - 1:
            neighbourhood.append((p[0] + 1, p[1]))
        if p[1] - 1 >= 0:
            neighbourhood.append((p[0], p[1] - 1))
        if p[1] + 1 < w.shape[1]:
            neighbourhood.append((p[0], p[1] + 1))

        for x, y in neighbourhood:
            # тут вычисляется расстояние
            d_tmp = d[p] + np.abs(w[x, y] - w[p])
            if d[x, y] > d_tmp:
                d[x, y] = d_tmp
                q.put((d[x, y], (x, y)))

        v[p] = True

    return d
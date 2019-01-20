# import matplotlib
import numpy as np
# import matplotlib.pyplot as plt  # Framefork
# import seaborn as sns # # Framefork
# sns.set_style("dark")
# plt.rcParams['figure.figsize'] = 16, 12
import pandas as pd
from PIL import Image
from tqdm import tqdm_notebook
from skimage import transform
import itertools as it #from more_itertools import chunked
from sklearn.neighbors.kde import KernelDensity
import matplotlib.cm as cm
import queue
from skimage import morphology
import dlib
import cv2
from imutils import face_utils
from scipy.spatial import Delaunay

from utils import dijkstra

# start_input = np.array(Image.open('maxim-min.png'))
img_input = np.array(Image.open('maxim-min.png'))
# img_input = cv2.resize(start_input, dsize=(100, 100), interpolation=cv2.INTER_CUBIC)

# инстанцируем класс для детекции лиц (рамка)
detector = dlib.get_frontal_face_detector()
# инстанцируем класс для детекции ключевых точек
predictor = dlib.shape_predictor('../data/shape_predictor_68_face_landmarks.dat')

# конвертируем изображение в много оттенков серого
img_gray = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY)
# вычисляем список рамок на каждое найденное лицо
rects = detector(img_gray, 0)
# вычисляем ключевые точки
shape = predictor(img_gray, rects[0])
shape = face_utils.shape_to_np(shape)

img_tmp = img_input.copy()
for x, y in shape:
    cv2.circle(img_tmp, (x, y), 1, (0, 0, 255), -1)
cv2.imwrite('face-points.jpg', img_tmp)

# оригинальная рамка
face_origin = sorted([(t.width()*t.height(), (t.left(), t.top(), t.width(), t.height())) for t in rects], key=lambda t: t[0], reverse=True)[0][1]

# коэффициенты расширения рамки
rescale = (1.3, 2.2, 1.3, 1.3)
# расширение рамки, так чтобы она не вылезла за края
(x, y, w, h) = face_origin
cx = x + w/2
cy = y + h/2
w = min(img_input.shape[1] - x, int(w/2 + rescale[2]*w/2))
h = min(img_input.shape[0] - y, int(h/2 + rescale[3]*h/2))

fx = max(0, int(x + w/2*(1 - rescale[0])))
fy = max(0, int(y + h/2*(1 - rescale[1])))
fw = min(img_input.shape[1] - fx, int(w - w/2*(1 - rescale[0])))
fh = min(img_input.shape[0] - fy, int(h - h/2*(1 - rescale[1])))

face = (fx, fy, fw, fh)

img_tmp = cv2.rectangle(img_input.copy(), (face[0], face[1]), (face[0] + face[2], face[1] + face[3]), (255, 0, 0), thickness=3, lineType=8, shift=0)
img_tmp = cv2.rectangle(img_tmp, (face_origin[0], face_origin[1]), (face_origin[0] + face_origin[2], face_origin[1] + face_origin[3]), (0, 255, 0), thickness=3, lineType=8, shift=0)

cv2.imwrite('face-points2.jpg', img_tmp)


# выбираем вышеописанные пять точек
points = [shape[0].tolist(), shape[16].tolist()]
for ix in [4, 12, 8]:
    x, y = shape[ix].tolist()
    points.append((x, y))
    points.append((x, points[0][1] + points[0][1] - y))

# img_tmp = img_input.copy()
# for x, y in points:
#     cv2.circle(img_tmp, (x, y), 2, (0, 0, 255), -1)
# cv2.imwrite('face-points3.jpg', img_tmp)

# я не особо в прототипе запариваюсь над производительностью
# так что вызываю триангуляцию Делоне,
# чтобы использовать ее как тест на то, что точка внутри полигона
# все это можно делать быстрее, т.к. точный тест не нужен
# для прототипа :good-enough: 
hull = Delaunay(points)
xy_fg = []
for x, y in it.product(range(img_input.shape[0]), range(img_input.shape[1])):
    if hull.find_simplex([y, x]) >= 0:
        xy_fg.append((x, y))
print('xy_fg%:', len(xy_fg)/np.prod(img_input.shape))

# вычисляем количество точек для фона
# примерно равно что бы было тому, что на лице
r = face[1]*face[3]/np.prod(img_input.shape[:2])
print(r)
k = 0.1
xy_bg_n = int(k*np.prod(img_input.shape[:2]))
print(xy_bg_n)

# накидываем случайные точки
xy_bg = zip(np.random.uniform(0, img_input.shape[0], size=xy_bg_n).astype(np.int),
            np.random.uniform(0, img_input.shape[1], size=xy_bg_n).astype(np.int))
xy_bg = list(xy_bg)
xy_bg = [(x, y) for (x, y) in xy_bg 
         if y < face[0] or y > face[0] + face[2] or x < face[1] or x > face[1] + face[3]]
print(len(xy_bg)/np.prod(img_input.shape[:2]))

# img_tmp = img_input.copy()
# for x, y in xy_fg:
#     img_tmp[x, y, :] = img_tmp[x, y, :]*0.5 + np.array([1, 0, 0]) * 0.5

# for x, y in xy_bg:
#     img_tmp[x, y, :] = img_tmp[x, y, :]*0.5 + np.array([0, 0, 1]) * 0.5
    
# cv2.imwrite('face-points5.jpg', img_tmp)

points_fg = np.array([img_input[x, y, :] for (x, y) in xy_fg])
points_bg = np.array([img_input[x, y, :] for (x, y) in xy_bg])


# инстанцируем классы KDE для объекта и фона
kde_fg = KernelDensity(kernel='gaussian', bandwidth=1, algorithm='kd_tree', leaf_size=100).fit(points_fg)
kde_bg = KernelDensity(kernel='gaussian', bandwidth=1, algorithm='kd_tree', leaf_size=100).fit(points_bg)

# инициализируем и вычисляем маски
score_kde_fg = np.zeros(img_input.shape[:2]) # заполняем нулями свежие матрицы
score_kde_bg = np.zeros(img_input.shape[:2])
likelihood_fg = np.zeros(img_input.shape[:2])
coodinates = it.product(range(score_kde_fg.shape[0]), range(score_kde_fg.shape[1]))

value = len(tqdm_notebook(coodinates, total=np.prod(score_kde_fg.shape)))

for x, y in tqdm_notebook(coodinates, total=np.prod(score_kde_fg.shape)):
    score_kde_fg[x, y] = np.exp(kde_fg.score(img_input[x, y, :].reshape(1, -1)))
    score_kde_bg[x, y] = np.exp(kde_bg.score(img_input[x, y, :].reshape(1, -1)))
    n = score_kde_fg[x, y] + score_kde_bg[x, y]
    if n == 0:
        n = 1
    likelihood_fg[x, y] = score_kde_fg[x, y]/n

print('Finish!')

# вызываем алгоритм для двух масок
d_fg = dijkstra(xy_fg, likelihood_fg)
d_bg = dijkstra(xy_bg, 1 - likelihood_fg)

print('Finish 2 !')

margin = 1.0
mask = (d_fg < (d_bg + margin)).astype(np.uint8)

cv2.imwrite('face-points6.jpg', mask)

img_fg = img_input.copy()
img_bg = (np.array(Image.open('max.jpg'))/255.0)[:800, :800, :]

x = int(img_bg.shape[0] - img_fg.shape[0])
y = int(img_bg.shape[1]/2 - img_fg.shape[1]/2)

img_bg_fg = img_bg[x:(x + img_fg.shape[0]), y:(y + img_fg.shape[1]), :]
mask_3d = np.dstack([mask, mask, mask])
img_bg[x:(x + img_fg.shape[0]), y:(y + img_fg.shape[1]), :] = mask_3d*img_fg + (1 - mask_3d)*img_bg_fg

cv2.imwrite('face-points7.jpg', img_bg)
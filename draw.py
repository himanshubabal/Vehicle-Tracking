from predict import draw_boxes, slide_window

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
import pickle
from sklearn.svm import LinearSVC, SVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog


image = mpimg.imread('test_images/test1.png')
draw_image = np.copy(image)

# y_start_stop = [350, 675]

y_start_stop = [350, 350 + 48 * 2]
xy_window = (24 * 2, 24 * 2)
windows = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop,
                       xy_window=xy_window, xy_overlap=(0.5, 0.5))
window_img = draw_boxes(draw_image, windows, color=(0, 0, 255), thick=6)


y_start_stop = [350 + 48, 350 + 48 * 5]
xy_window = (24 * 3, 24 * 3)
windows = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop,
                       xy_window=xy_window, xy_overlap=(0.5, 0.5))
window_img = draw_boxes(window_img, windows, color=(255, 0, 0), thick=6)

y_start_stop = [350 + 48 * 2, 350 + 48 * 6]
xy_window = (24 * 4, 24 * 4)
windows = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop,
                       xy_window=xy_window, xy_overlap=(0.5, 0.5))
window_img = draw_boxes(window_img, windows, color=(255, 0, 0), thick=6)

y_start_stop = [350 + 48 * 3, 350 + 48 * 7]
xy_window = (24 * 5, 24 * 5)
windows = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop,
                       xy_window=xy_window, xy_overlap=(0.5, 0.5))
window_img = draw_boxes(window_img, windows, color=(255, 0, 0), thick=6)

y_start_stop = [350 + 48 * 3, 350 + 48 * 7]
xy_window = (24 * 6, 24 * 6)
windows = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop,
                       xy_window=xy_window, xy_overlap=(0.5, 0.5))
window_img = draw_boxes(window_img, windows, color=(255, 0, 0), thick=6)

y_start_stop = [350 + 48 * 3, 350 + 48 * 7]
xy_window = (24 * 7, 24 * 7)
windows = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop,
                       xy_window=xy_window, xy_overlap=(0.5, 0.5))
window_img = draw_boxes(window_img, windows, color=(255, 0, 0), thick=6)

plt.imshow(window_img)

plt.show()

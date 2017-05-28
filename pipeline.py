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

from sklearn.model_selection import train_test_split
from scipy.ndimage.measurements import label

from classify import get_hog_features, bin_spatial, extract_features
from classify import color_hist, get_saved_SVM, get_train_data

from predict import draw_boxes, add_heat, apply_threshold, draw_labeled_bboxes, find_cars

from utility import get_data
import utility


def pipeline(X_scaler, svc, img, ystart=400, ystop=656, scale=1.25):
    # img = mpimg.imread(image)
    color_space = utility.color_space  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    orient = utility.orient  # HOG orientations
    pix_per_cell = utility.pix_per_cell  # HOG pixels per cell
    cell_per_block = utility.cell_per_block  # HOG cells per block
    hog_channel = utility.hog_channel  # Can be 0, 1, 2, or "ALL"

    spatial_size = utility.spatial_size  # Spatial binning dimensions
    hist_bins = utility.hist_bins    # Number of histogram bins
    spatial_feat = utility.spatial_feat  # Spatial features on or off
    hist_feat = utility.hist_feat  # Histogram features on or off
    hog_feat = utility.hog_feat  # HOG features on or off


    out_img, bbox_list = find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, hog_channel)

    heat = np.zeros_like(img[:, :, 0]).astype(np.float)

    # Add heat to each box in box list
    heat = add_heat(heat, bbox_list)

    # Apply threshold to help remove false positives
    heat = apply_threshold(heat, 1)

    # Visualize the heatmap when displaying
    heatmap = np.clip(heat, 0, 255)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(np.copy(img), labels)

    return draw_img

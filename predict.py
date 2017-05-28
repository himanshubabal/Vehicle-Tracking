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
from utility import get_data
import utility
# Define a function to draw bounding boxes


def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy


# Define a function that takes an image,
# start and stop positions in both x and y,
# window size (x and y dimensions),
# and overlap fraction (for both x and y)
# def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None],
#                  xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
#     # If x and/or y start/stop positions not defined, set to image size
#     if x_start_stop[0] == None:
#         x_start_stop[0] = 0
#     if x_start_stop[1] == None:
#         x_start_stop[1] = img.shape[1]
#     if y_start_stop[0] == None:
#         y_start_stop[0] = 0
#     if y_start_stop[1] == None:
#         y_start_stop[1] = img.shape[0]
#     # Compute the span of the region to be searched
#     xspan = x_start_stop[1] - x_start_stop[0]
#     yspan = y_start_stop[1] - y_start_stop[0]
#     # Compute the number of pixels per step in x/y
#     nx_pix_per_step = np.int(xy_window[0] * (1 - xy_overlap[0]))
#     ny_pix_per_step = np.int(xy_window[1] * (1 - xy_overlap[1]))
#     # Compute the number of windows in x/y
#     nx_buffer = np.int(xy_window[0] * (xy_overlap[0]))
#     ny_buffer = np.int(xy_window[1] * (xy_overlap[1]))
#     nx_windows = np.int((xspan - nx_buffer) / nx_pix_per_step)
#     ny_windows = np.int((yspan - ny_buffer) / ny_pix_per_step)
#     # Initialize a list to append window positions to
#     window_list = []
#     # Loop through finding x and y window positions
#     # Note: you could vectorize this step, but in practice
#     # you'll be considering windows one by one with your
#     # classifier, so looping makes sense
#     for ys in range(ny_windows):
#         for xs in range(nx_windows):
#             # Calculate window position
#             startx = xs * nx_pix_per_step + x_start_stop[0]
#             endx = startx + xy_window[0]
#             starty = ys * ny_pix_per_step + y_start_stop[0]
#             endy = starty + xy_window[1]

#             # Append window position to list
#             window_list.append(((startx, starty), (endx, endy)))
#     # Return the list of windows
#     return window_list

# Define a function to extract features from a single image window
# This function is very similar to extract_features()
# just for a single image rather than list of images
# def single_img_features(img, color_space='RGB', spatial_size=(32, 32),
#                         hist_bins=32, orient=9,
#                         pix_per_cell=8, cell_per_block=2, hog_channel=0,
#                         spatial_feat=True, hist_feat=True, hog_feat=True):
#     # 1) Define an empty list to receive features
#     img_features = []
#     # 2) Apply color conversion if other than 'RGB'
#     if color_space != 'RGB':
#         if color_space == 'HSV':
#             feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
#         elif color_space == 'LUV':
#             feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
#         elif color_space == 'HLS':
#             feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
#         elif color_space == 'YUV':
#             feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
#         elif color_space == 'YCrCb':
#             feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
#     else:
#         feature_image = np.copy(img)
#     # 3) Compute spatial features if flag is set
#     if spatial_feat == True:
#         spatial_features = bin_spatial(feature_image, size=spatial_size)
#         # 4) Append features to list
#         img_features.append(spatial_features)
#     # 5) Compute histogram features if flag is set
#     if hist_feat == True:
#         hist_features = color_hist(feature_image, nbins=hist_bins)
#         # 6) Append features to list
#         img_features.append(hist_features)
#     # 7) Compute HOG features if flag is set
#     if hog_feat == True:
#         if hog_channel == 'ALL':
#             hog_features = []
#             for channel in range(feature_image.shape[2]):
#                 hog_features.extend(get_hog_features(feature_image[:, :, channel],
#                                                      orient, pix_per_cell, cell_per_block,
#                                                      vis=False, feature_vec=True))
#         else:
#             hog_features = get_hog_features(feature_image[:, :, hog_channel], orient,
#                                             pix_per_cell, cell_per_block, vis=False, feature_vec=True)
#         # 8) Append features to list
#         img_features.append(hog_features)

#     # 9) Return concatenated array of features
#     return np.concatenate(img_features)


# def search_windows(img, windows, clf, scaler, color_space='RGB',
#                    spatial_size=(32, 32), hist_bins=32,
#                    hist_range=(0, 256), orient=9,
#                    pix_per_cell=8, cell_per_block=2,
#                    hog_channel=0, spatial_feat=True,
#                    hist_feat=True, hog_feat=True):

#     # 1) Create an empty list to receive positive detection windows
#     on_windows = []
#     # 2) Iterate over all windows in the list
#     for window in windows:
#         # 3) Extract the test window from original image
#         test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
#         # 4) Extract features for that window using single_img_features()
#         features = single_img_features(test_img, color_space=color_space,
#                                        spatial_size=spatial_size, hist_bins=hist_bins,
#                                        orient=orient, pix_per_cell=pix_per_cell,
#                                        cell_per_block=cell_per_block,
#                                        hog_channel=hog_channel, spatial_feat=spatial_feat,
#                                        hist_feat=hist_feat, hog_feat=hog_feat)
#         # 5) Scale extracted features to be fed to classifier
#         test_features = scaler.transform(np.array(features).reshape(1, -1))
#         # 6) Predict using your classifier
#         prediction = clf.predict(test_features)
#         # 7) If positive (prediction == 1) then save the window
#         if prediction == 1:
#             on_windows.append(window)
#     # 8) Return windows for positive detections
#     return on_windows

def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    # Iterate through list of bboxes
    return heatmap


def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap


def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1] + 1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)
    # Return the image
    return img



def find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, hog_channel):

    draw_img = np.copy(img)
    # img = img.astype(np.float32)/255

    img_tosearch = img[ystart:ystop,:,:]
    ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2YUV)
    # ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))

    # Compute individual channel HOG features for the entire image
    if hog_channel == 'ALL':
        ch1 = ctrans_tosearch[:,:,0]
        ch2 = ctrans_tosearch[:,:,1]
        ch3 = ctrans_tosearch[:,:,2]

        hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

    else:
        ch = ctrans_tosearch[:,:,hog_channel]
        hog = get_hog_features(ch, orient, pix_per_cell, cell_per_block, feature_vec=False)

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1
    nfeat_per_block = orient*cell_per_block**2

    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    bbox_list = list()
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step

            # Extract HOG for this patch
            if hog_channel == 'ALL':
                hog_feat1 = hog1[ypos : ypos +nblocks_per_window,
                                    xpos:xpos+nblocks_per_window].ravel()
                hog_feat2 = hog2[ypos : ypos + nblocks_per_window,
                                    xpos:xpos+nblocks_per_window].ravel()
                hog_feat3 = hog3[ypos : ypos + nblocks_per_window,
                                    xpos:xpos+nblocks_per_window].ravel()
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            else:
                hog_features = hog[ypos : ypos + nblocks_per_window,
                                    xpos : xpos + nblocks_per_window].ravel()

            xleft = xpos * pix_per_cell
            ytop = ypos * pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))

            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)


            hstack = np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1)
            test_features = X_scaler.transform(hstack)
            #test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))
            test_prediction = svc.predict(test_features)

            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)

                bbox_list.append(((xbox_left, ytop_draw+ystart), (xbox_left+win_draw,ytop_draw+win_draw+ystart)))

                cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),(0,0,255),6)

    return (draw_img, bbox_list)


if __name__ == '__main__':
    # sample_size = -1    # Use all trainig examples
    # cars, notcars = get_data(sample_size=sample_size)
    # X_train, y_train, X_test, y_test, X_scaler = get_train_data(cars, notcars, verbrose=True)
    t = time.time()
    with open('data/X_scaler.pkl', 'rb') as file:
        # X = pickle.load(file)
        X_scaler = pickle.load(file)

    print('X pkl time : ', time.time()-t)

    # X_scaler = StandardScaler().fit(X)

    svc = get_saved_SVM(verbrose=True)

    img_list = ['test_images/test1.png', 'test_images/test2.png', 'test_images/test3.png',
                'test_images/test4.png', 'test_images/test5.png']

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

    ystart = 400
    ystop = 656
    scale = 1.25

    for image in img_list:
        t = time.time()
        img = mpimg.imread(image)
        out_img, bbox_list = find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, hog_channel)

        print('Time : ', time.time()-t)
        plt.imshow(out_img)
        plt.show()

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

        fig = plt.figure()
        plt.subplot(121)
        plt.imshow(draw_img)
        plt.title('Car Positions')
        plt.subplot(122)
        plt.imshow(heatmap, cmap='hot')
        plt.title('Heat Map')
        fig.tight_layout()
        plt.show()

    # for img in img_list:
    #     # Predict
    #     image = mpimg.imread(img)
    #     draw_image = np.copy(image)

    #     t1 = time.time()
    #     # Uncomment the following line if you extracted training
    #     # data from .png images (scaled 0 to 1 by mpimg) and the
    #     # image you are searching is a .jpg (scaled 0 to 255)
    #     # image = image.astype(np.float32)/255
    #     color_space = 'YUV'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    #     orient = 12  # HOG orientations
    #     pix_per_cell = 8  # HOG pixels per cell
    #     cell_per_block = 3  # HOG cells per block
    #     hog_channel = 0  # Can be 0, 1, 2, or "ALL"

    #     spatial_size = (16, 16)  # Spatial binning dimensions
    #     hist_bins = 16    # Number of histogram bins
    #     spatial_feat = True  # Spatial features on or off
    #     hist_feat = True  # Histogram features on or off
    #     hog_feat = True  # HOG features on or off

    #     # y_start_stop = [None, None]  # Min and max in y to search in slide_window()
    #     # y_start_stop = [350, 675]

    #     # y_start_stop = [350, 350 + 96 * 2]

    #     # xy_window = (96, 96)
    #     # windows = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop,
    #     #                        xy_window=xy_window, xy_overlap=(0.5, 0.5))
    #     # window_img = draw_boxes(draw_image, windows, color=(0, 0, 255), thick=6)

    #     # y_start_stop = [350 + 96 / 2, 350 + 96 * 3]   # ()
    #     # xy_window = (144, 144)
    #     # windows = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop,
    #     #                        xy_window=xy_window, xy_overlap=(0.5, 0.5))
    #     # window_img = draw_boxes(window_img, windows, color=(255, 0, 0), thick=6)

    #     # y_start_stop = [350 + (96 / 2) * 3, 700]   # ()
    #     # xy_window = (192, 192)
    #     # windows = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop,
    #     #                        xy_window=xy_window, xy_overlap=(0.5, 0.5))
    #     # window_img = draw_boxes(window_img, windows, color=(255, 0, 0), thick=6)

    #     window_list = []
    #     # # # # # # # # # # # # # # # # # # # # # #
    #     # y_start_stop = [350, 350 + 96]
    #     # xy_window = (96 / 2, 96 / 2)
    #     y_start_stop = [350, 350 + 48 * 2]
    #     xy_window = (24 * 2, 24 * 2)
    #     windows = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop,
    #                            xy_window=xy_window, xy_overlap=(0.5, 0.5))
    #     for w in windows:
    #         window_list.append(w)
    #     # # # # # # # # # # # # # # # # # # # # # #
    #     # y_start_stop = [350, 350 + 96 * 2]
    #     # xy_window = (96, 96)
    #     y_start_stop = [350 + 48, 350 + 48 * 5]
    #     xy_window = (24 * 3, 24 * 3)
    #     windows = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop,
    #                            xy_window=xy_window, xy_overlap=(0.5, 0.5))
    #     for w in windows:
    #         window_list.append(w)
    #     # # # # # # # # # # # # # # # # # # # # # #
    #     # y_start_stop = [350 + 96 / 2, 350 + 96 * 3]
    #     # xy_window = (144, 144)
    #     y_start_stop = [350 + 48 * 2, 350 + 48 * 6]
    #     xy_window = (24 * 4, 24 * 4)
    #     windows = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop,
    #                            xy_window=xy_window, xy_overlap=(0.5, 0.5))
    #     for w in windows:
    #         window_list.append(w)
    #     # # # # # # # # # # # # # # # # # # # # # #
    #     # y_start_stop = [350 + 96, 350 + 96 * 3]
    #     # xy_window = (128, 128)
    #     y_start_stop = [350 + 48 * 3, 350 + 48 * 7]
    #     xy_window = (24 * 5, 24 * 5)
    #     windows = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop,
    #                            xy_window=xy_window, xy_overlap=(0.5, 0.5))
    #     for w in windows:
    #         window_list.append(w)
    #     # # # # # # # # # # # # # # # # # # # # # #
    #     # y_start_stop = [350 + 96 * 2, 350 + 96 * 5]
    #     # xy_window = (160, 160)
    #     y_start_stop = [350 + 48 * 3, 350 + 48 * 7]
    #     xy_window = (24 * 6, 24 * 6)
    #     windows = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop,
    #                            xy_window=xy_window, xy_overlap=(0.5, 0.5))
    #     for w in windows:
    #         window_list.append(w)
    #     # # # # # # # # # # # # # # # # # # # # # #
    #     # y_start_stop = [350 + 96 * 2, 350 + 96 * 4]
    #     # xy_window = (160, 160)
    #     y_start_stop = [350 + 48 * 3, 350 + 48 * 7]
    #     xy_window = (24 * 7, 24 * 7)
    #     windows = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop,
    #                            xy_window=xy_window, xy_overlap=(0.5, 0.5))
    #     for w in windows:
    #         window_list.append(w)
    #     # # # # # # # # # # # # # # # # # # # # # #
    #     # y_start_stop = [350 + 144, 700]
    #     # xy_window = (192, 192)
    #     # windows = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop,
    #                            # xy_window=xy_window, xy_overlap=(0.5, 0.5))
    #     # for w in windows:
    #         # window_list.append(w)
    #     # # # # # # # # # # # # # # # # # # # # # #
    #     len(window_list)
    #     hot_windows = search_windows(image, window_list, svc, X_scaler, color_space=color_space,
    #                                  spatial_size=spatial_size, hist_bins=hist_bins,
    #                                  orient=orient, pix_per_cell=pix_per_cell,
    #                                  cell_per_block=cell_per_block,
    #                                  hog_channel=hog_channel, spatial_feat=spatial_feat,
    #                                  hist_feat=hist_feat, hog_feat=hog_feat)

    #     window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)

    #     print('Time taken : ', time.time() - t1)
    #     plt.imshow(window_img)
    #     plt.show()

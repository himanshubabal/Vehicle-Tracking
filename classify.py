import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
import pickle
import os
from sklearn.svm import LinearSVC, SVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.model_selection import train_test_split, GridSearchCV

from utility import get_data
import utility


def get_hog_features(img, orient, pix_per_cell, cell_per_block,
                     vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True,
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True,
                       visualise=vis, feature_vector=feature_vec)
        return features


# Define a function to compute binned color features
def bin_spatial(img, size=(32, 32)):
    color1 = cv2.resize(img[:, :, 0], size).ravel()
    color2 = cv2.resize(img[:, :, 1], size).ravel()
    color3 = cv2.resize(img[:, :, 2], size).ravel()
    return np.hstack((color1, color2, color3))


# Define a function to compute color histogram features
# NEED TO CHANGE bins_range if reading .png files with mpimg!
# bin_range -> .jpg = (0, 256)  and  -> .png = (0, 1)
def color_hist(img, nbins=32, bins_range=(0, 1)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:, :, 0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:, :, 1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:, :, 2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features


def extract_features(imgs, color_space='RGB', spatial_size=(32, 32),
                     hist_bins=32, orient=9,
                     pix_per_cell=8, cell_per_block=2, hog_channel=0,
                     spatial_feat=True, hist_feat=True, hog_feat=True, verbrose=False):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        file_features = []
        # Read in each one by one
        image = mpimg.imread(file)
        # apply color conversion if other than 'RGB'
        if color_space != 'RGB':
            if color_space == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif color_space == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif color_space == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif color_space == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif color_space == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCR_CB)
        else:
            feature_image = np.copy(image)

        if spatial_feat == True:
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            if verbrose:
                print('spatial bin : ', spatial_features.shape)
            file_features.append(spatial_features)
        if hist_feat == True:
            # Apply color_hist()
            hist_features = color_hist(feature_image, nbins=hist_bins)
            if verbrose:
                print('hist features : ', hist_features.shape)
            file_features.append(hist_features)
        if hog_feat == True:
            # Call get_hog_features() with vis=False, feature_vec=True
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(get_hog_features(feature_image[:, :, channel],
                                                         orient, pix_per_cell, cell_per_block,
                                                         vis=False, feature_vec=True))
                hog_features = np.ravel(hog_features)
            else:
                hog_features = get_hog_features(feature_image[:, :, hog_channel], orient,
                                                pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            # Append the new feature vector to the features list
            if verbrose:
                print('hog bin : ', hog_features.shape)
            file_features.append(hog_features)
        features.append(np.concatenate(file_features))
    # Return list of feature vectors
    return features


def get_train_data(cars, notcars, verbrose=False, save_data=True):
    # BEST HYPER PARAMETERS

    # Extrace Features
    # color_space = 'YUV'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    # orient = 9  # HOG orientations
    # pix_per_cell = 8  # HOG pixels per cell
    # cell_per_block = 2  # HOG cells per block
    # hog_channel = 0  # Can be 0, 1, 2, or "ALL"

    # spatial_size = (16, 16)  # Spatial binning dimensions
    # hist_bins = 16    # Number of histogram bins
    # spatial_feat = True  # Spatial features on or off
    # hist_feat = True  # Histogram features on or off
    # hog_feat = True  # HOG features on or off

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

    car_features = extract_features(cars, color_space=color_space,
                                    spatial_size=spatial_size, hist_bins=hist_bins,
                                    orient=orient, pix_per_cell=pix_per_cell,
                                    cell_per_block=cell_per_block,
                                    hog_channel=hog_channel, spatial_feat=spatial_feat,
                                    hist_feat=hist_feat, hog_feat=hog_feat)

    notcar_features = extract_features(notcars, color_space=color_space,
                                       spatial_size=spatial_size, hist_bins=hist_bins,
                                       orient=orient, pix_per_cell=pix_per_cell,
                                       cell_per_block=cell_per_block,
                                       hog_channel=hog_channel, spatial_feat=spatial_feat,
                                       hist_feat=hist_feat, hog_feat=hog_feat)

    X = np.vstack((car_features, notcar_features)).astype(np.float64)
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    rand_state = np.random.randint(0, 100)
    # Split in train-test data
    X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2,
                                                        random_state=rand_state)

    if verbrose:
        print('Using:', orient, 'orientations', pix_per_cell,
              'pixels per cell and', cell_per_block, 'cells per block')
        print('Feature vector length:', len(X_train[0]))

        print(X.shape, y.shape)
        print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    # Saving Data
    if save_data:
        print('Saving Data')
        # with open('data/X_train.pkl', 'wb') as file:
        #     pickle.dump(X_train, file)

        # with open('data/X_test.pkl', 'wb') as file:
        #     pickle.dump(X_test, file)

        # with open('data/y_train.pkl', 'wb') as file:
        #     pickle.dump(y_train, file)

        # with open('data/y_test.pkl', 'wb') as file:
        #     pickle.dump(y_test, file)

        with open('data/X_all.pkl', 'wb') as file:
            pickle.dump(X, file)

    return(X_train, y_train, X_test, y_test, X_scaler)


def train_SVM(X_train, y_train, X_test, y_test, save_model=True):
    # BEST SVM HYPERPARAMETERS
    C = 1
    kernel = 'rbf'
    gamma = 1e-5

    # Hyper-Param tuning
    # param_grid = [{'C': [1, 10], 'kernel': ['linear']},
    #               {'C': [1, 10], 'gamma': [0.001, 0.0001, 0.00001], 'kernel': ['rbf']}, ]

    # svc = SVC()
    # clf = GridSearchCV(svc, param_grid, verbose=10)
    # clf.fit(X_train, y_train)
    # best_param = clf.best_params_
    # print('Best parameters : ', best_param)

    # C = best_param['C']
    # kernel = best_param['kernel']
    # if kernel == 'rbf':
    #     gamma = best_param['gamma']
    # else:
    #     gamma = 'auto'
    print('Training SVM')
    svc = SVC(C=C, kernel=kernel, gamma=gamma, verbose=True)
    svc.fit(X_train, y_train)

    # Save model
    if save_model:
        with open('saved_models/svm_all.pkl', 'wb') as fid:
            pickle.dump(svc, fid)

    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))

    return(svc)


def get_saved_SVM(verbrose=False):
    if not os.path.isfile('saved_models/svm_all.pkl'):
        if verbrose:
            print('Saved model not found, training SVM now')

        cars, notcars = get_data(sample_size=-1)
        X_train, y_train, X_test, y_test = get_train_data(cars, notcars)
        train_SVM(X_train, y_train, X_test, y_test)

    if verbrose:
        print('Loading the saved model')
    # load the model
    with open('saved_models/svm_all.pkl', 'rb') as file:
        svc = pickle.load(file)

    return svc


def predict_SVM(X_pred, verbrose=False):
    if not os.path.isfile('saved_models/svm_all.pkl'):
        if verbrose:
            print('Saved model not found, training SVM now')

        cars, notcars = get_data(sample_size=-1)
        X_train, y_train, X_test, y_test = get_train_data(cars, notcars)
        train_SVM(X_train, y_train, X_test, y_test)

    if verbrose:
        print('Loading the saved model')
    # load the model
    with open('saved_models/svm_all.pkl', 'rb') as file:
        svc = pickle.load(file)

    # Return the predicted result
    return(svc.predict(X_pred))

if __name__ == '__main__':
    # Divide up into cars and notcars
    sample_size = -1    # Use all trainig examples
    cars, notcars = get_data(sample_size=sample_size)
    X_train, y_train, X_test, y_test, X_scaler = get_train_data(cars, notcars, verbrose=True)
    train_SVM(X_train, y_train, X_test, y_test)

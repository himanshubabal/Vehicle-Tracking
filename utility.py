import fnmatch
import os

color_space = 'YCrCb'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8  # HOG pixels per cell
cell_per_block = 2  # HOG cells per block
hog_channel = 'ALL'  # Can be 0, 1, 2, or "ALL"

spatial_size = (32, 32)  # Spatial binning dimensions
hist_bins = 32    # Number of histogram bins
spatial_feat = True  # Spatial features on or off
hist_feat = True  # Histogram features on or off
hog_feat = True  # HOG features on or off


def get_matches(dir='data', pattern='.png'):
    matches = []
    for root, dirnames, filenames in os.walk(dir):
        for filename in fnmatch.filter(filenames, pattern):
            matches.append(os.path.join(root, filename))

    return matches


def get_data(sample_size=-1):
    images_cars = get_matches(dir='data/vehicles', pattern='*.png')
    images_notcars = get_matches(dir='data/non_vehicles', pattern='*.png')

    cars, notcars = [], []

    if sample_size > min(len(images_cars), len(images_notcars)):
        raise ValueError('length can not be more then list')

    if sample_size == -1:
        for image in images_cars:
            cars.append(image)

        for image in images_notcars:
            notcars.append(image)

    else:
        for image in images_cars[:sample_size]:
            cars.append(image)

        for image in images_notcars[:sample_size]:
            notcars.append(image)

    return(cars, notcars)

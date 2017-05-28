## Writeup

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/Figure_1.jpg
[image4]: ./examples/Figure_1-4.jpg
[image8]: ./examples/Figure_1-6.jpg

[image5]: ./examples/Figure_1-3.jpg
[image6]: ./examples/Figure_1-5.jpg
[image7]: ./examples/Figure_1-7.jpg
[video1]: ./project_video_out.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained method `get_hog_features()` from line 18 to line 31 in file `classify.py`

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  

After experimenting with different values, I settled on these values for the HOG parameters :
```python
orient = 9  # HOG orientations
pix_per_cell = 8  # HOG pixels per cell
cell_per_block = 2  # HOG cells per block
hog_channel = 'ALL'  # Can be 0, 1, 2, or "ALL"
```

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and the above mentioned 3 seemed to be giving best accuracy in the SVM classifier

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

file name `classify.py`, method `train_SVM()`  line 199 to line 233.


I used HOG Features as well as Color features and those lead to a total of `8460` features.

I used `GridSearchCV()` with the following parameters -

```python
Hyper-Param tuning
param_grid = [{'C': [1, 10, 100], 'kernel': ['linear']},
              {'C': [1, 10, 100], 'gamma': [0.001, 0.0001, 0.00001],
              'kernel': ['rbf']}, ]

svc = SVC()
clf = GridSearchCV(svc, param_grid, verbose=10)
clf.fit(X_train, y_train)
best_param = clf.best_params_
print('Best parameters : ', best_param)
```

The best parameters turned out to be `C : 1, kernel : rbf, gamma : 1e-4` with test accuracy of more then `97%`.


### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?


Method `find_cars()` in `predict.py` from line 195 to line 278.

I searched only a specific region in image where I knew there was a possibility to find cars.
It helped me to reduce time and area of search.

Then I implemented sliding window to search image and classify it as true or false.

Also, I extracted HOG features for each image just once, so I need not to do it for every window, it saved me bunch of time.

My images had 50% overlap and varied in sizes.
Final output looked something like this -

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
![alt text][image8]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Video can be found in project folder and is named `project_video_out.mp4`


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and final bounding box obtained.


![alt text][image5]
![alt text][image6]
![alt text][image7]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

My pipeline did take a lot of time (nearly 15 seconds per frame), thus I think implementing it in real-world scenario will be a bit difficult.


Also, I would like to do this video through Deep Learning approach.

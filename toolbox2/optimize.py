# This is a routine to try to test postprocessing against two sets of images.
# It takes two folders, one full of manually thresholded images, another of unthresholded images
# Compares them, builds up some error numbers and tries to make it easy to develop the 'hard bit' of this problem

# Parameters to play with:
# Threshold - What it says on the tin
# Adaptive Thresholding?
# Whether or not to do mean shift filtering
# Coefficients of mean shifting
# Gaussian blur, otsu's binarization

# What's the goal? Minimize mean squared error to the manually thresholded images.

import glob
import cv2
import numpy as np


def mse(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err

cropped_images = glob.glob('cropped/*.jpg')
manthresh_images = glob.glob('checkerboard/*.jpg')

MSEs = []

blursize = 3
threshold = 0 # When using otsu binarization, threshold is automatically determined

# all signs point to these being the best coefficients
thresholda = 50
thresholdb = 70
kernel_size = 7


for kernel_size in range(1,15,2):
#    for thresholda in range(50,200,20):
#            for thresholdb in range(thresholda,250,20):

        MSEs = []

        for cropped_imagename in cropped_images:
            #print("processing: ", cropped_imagename)

            _, imagename = cropped_imagename.split('/')

            cropped_image = cv2.imread(cropped_imagename)
        #    cv2.imshow("Cropped Input", cropped_image)

            # Read in manually thresholded image

            manual_imagename = "manual_thresh/" + imagename

            manual_image = cv2.imread(manual_imagename)
        #    cv2.imshow("Manual Input", manual_image)
            manual_image = cv2.cvtColor(manual_image, cv2.COLOR_BGR2GRAY)

            # Redo the threshold, just to make sure it's in the right colour space etc
            retval, manual_image_threshold = cv2.threshold(manual_image, 50, 255, cv2.THRESH_BINARY_INV)

            manual_image_cont, manual_contours, manual_hierarchy = cv2.findContours(manual_image_threshold,
                                                                                       cv2.RETR_EXTERNAL,
                                                                                       cv2.CHAIN_APPROX_NONE)
            largest_area = 0
            largest_contour_index = 0

            for index, contour in enumerate(manual_contours):
                area = cv2.contourArea(contour)
                if (area > largest_area):
                    largest_area = area
                    largest_contour_index = index

            filled_manual_contours = np.zeros_like(manual_image_cont)
            cv2.drawContours(filled_manual_contours, manual_contours, largest_contour_index, 255, -1)
            cv2.imshow("Filled Manual Contour",filled_manual_contours)


            # This is where to do all the things
            #                cropped_image = cv2.pyrMeanShiftFiltering(cropped_image, mean_SP, mean_SR, mean_MaxLevel)

            kernel = np.ones((kernel_size,kernel_size),np.uint8)

            edges = cv2.Canny(cropped_image,thresholda,thresholdb)

        #    cv2.imshow("edges",edges)

        # edges = cv2.morphologyEx(edges,cv2.MORPH_OPEN,kernel)
            edges = cv2.morphologyEx(edges,cv2.MORPH_CLOSE,kernel)

        #    cv2.imshow("opened closed",edges)

            cropped_image = cv2.GaussianBlur(cropped_image, (blursize, blursize), 0)
            cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

         #   retval, cropped_image_threshold = cv2.threshold(cropped_image, threshold, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

            cropped_image_cont, cropped_contours, cropped_hierarchy = cv2.findContours(edges,
                                                                                       cv2.RETR_EXTERNAL,
                                                                                       cv2.CHAIN_APPROX_NONE)
            largest_area = 0
            largest_filled_contour_index = 0

            for index, contour in enumerate(cropped_contours):
                area = cv2.contourArea(contour)
                if (area > largest_area):
                    largest_area = area
                    largest_filled_contour_index = index

            filled_contours = np.zeros_like(cropped_image_cont)
            cv2.drawContours(filled_contours, cropped_contours, largest_filled_contour_index, 255, -1)
            cv2.imshow("Cropped Contour",filled_contours)

            # Stop doing all the things and measure
            current_MSE = mse(filled_contours, filled_manual_contours)
            MSEs.append(current_MSE)

            #print("MSE: ",current_MSE)
            cv2.imshow("Manual Image Thresholded", manual_image_threshold)
#            cv2.imshow("Cropped Image Thresholded", cropped_image_threshold)
            #cv2.waitKey(0)
        print("Thresholda",thresholda,"thresholdb",thresholdb,"kernelsize",kernel_size,"Sum MSE",sum(MSEs),"max mse",max(MSEs),"mean mse",np.mean(MSEs))






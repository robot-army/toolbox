from __future__ import print_function
from __future__ import division
import cv2 as cv
import numpy as np
import argparse
from math import atan2, cos, sin, sqrt, pi

#  Sort out the orientation of the image. Struggles with tape measure (sample image 7) 'cos its pretty round. Struggles
#  with image 10 because of the scaling error

def drawAxis(img, p_, q_, colour, scale):
    p = list(p_)
    q = list(q_)

    angle = atan2(p[1] - q[1], p[0] - q[0])  # angle in radians
    hypotenuse = sqrt((p[1] - q[1]) * (p[1] - q[1]) + (p[0] - q[0]) * (p[0] - q[0]))
    # Here we lengthen the arrow by a factor of scale
    q[0] = p[0] - scale * hypotenuse * cos(angle)
    q[1] = p[1] - scale * hypotenuse * sin(angle)
    cv.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 2, cv.LINE_AA)
    # create the arrow hooks
    p[0] = q[0] + 9 * cos(angle + pi / 4)
    p[1] = q[1] + 9 * sin(angle + pi / 4)
    cv.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 2, cv.LINE_AA)
    p[0] = q[0] + 9 * cos(angle - pi / 4)
    p[1] = q[1] + 9 * sin(angle - pi / 4)
    cv.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 2, cv.LINE_AA)

def getOrientation(pts, img):
    sz = len(pts)
    data_pts = np.empty((sz, 2), dtype=np.float64)
    for i in range(data_pts.shape[0]):
        data_pts[i, 0] = pts[i, 0, 0]
        data_pts[i, 1] = pts[i, 0, 1]
    # Perform PCA analysis
    mean = np.empty((0))
    mean, eigenvectors, eigenvalues = cv.PCACompute2(data_pts, mean)
    # Store the center of the object
    cntr = (int(mean[0, 0]), int(mean[0, 1]))

    cv.circle(img, cntr, 3, (255, 0, 255), 2)
    p1 = (
    cntr[0] + 0.02 * eigenvectors[0, 0] * eigenvalues[0, 0], cntr[1] + 0.02 * eigenvectors[0, 1] * eigenvalues[0, 0])
    p2 = (
    cntr[0] - 0.02 * eigenvectors[1, 0] * eigenvalues[1, 0], cntr[1] - 0.02 * eigenvectors[1, 1] * eigenvalues[1, 0])
    drawAxis(img, cntr, p1, (0, 255, 0), 0.05)
    drawAxis(img, cntr, p2, (255, 255, 0), 0.1)
    angle = atan2(eigenvectors[0, 1], eigenvectors[0, 0])  # orientation in radians

    return angle


def orientation(src):

    parser = argparse.ArgumentParser(description='Code for Introduction to Principal Component Analysis (PCA) tutorial.\
                                                  This program demonstrates how to use OpenCV PCA to extract the orientation of an object.')
    parser.add_argument('--input', help='Path to input image.', default='../data/pca_test1.jpg')
#    args = parser.parse_args()
#    src = cv.imread(args.input)
    # Check if image is loaded successfully
    if src is None:
        print('Could not open or find the image: ', args.input)
        exit(0)
   # cv.imshow('src', cv.resize(src, (0, 0), fx=0.1, fy=0.1))    # Convert image to grayscale
    try:
        gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
        _, bw = cv.threshold(gray, 50, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    except:
        bw = src
    # Convert image to binary
    _, contours, _ = cv.findContours(bw, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    for i, c in enumerate(contours):
        # Calculate the area of each contour
        area = cv.contourArea(c);
        # Ignore contours that are too small or too large
        if  area < 4000 or 600000 < area:
            continue
        # Draw each contour only for visualisation purposes
#        src = cv.cvtColor(src, cv.COLOR_GRAY2BGR)
        cv.drawContours(src, contours, i, (0, 0, 255), 2);
        print("drawing area ", area)
        # Find the orientation of each shape
        getOrientation(c, src)
    cv.imshow('output', src)
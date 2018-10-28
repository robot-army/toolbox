import cv2 as cv
from unwarp import four_point_transform

# This should fix the image to remove any camera distortion (perspective, skew, etc). This should be change to work with
# a checkerboard of fixed grid pitch.

def fixperspective(threshold_perspective):
    # find all the contours in the image
    img, contourPerspective, hierarchyPerspective = cv.findContours(threshold_perspective, cv.RETR_EXTERNAL,
                                                                    cv.CHAIN_APPROX_SIMPLE)

    # Find the largest rectangular contour in the image. This should error out if it doesn't find one
    # the rectangular contour is a ruler, 6" x 1" in the sample images. It's needed for scaling and deskewing

    largest_area = 0
    largestContourIndex = 0

    for index, val in enumerate(contourPerspective):
        contourPerspective[index] = cv.approxPolyDP(val, 13, True)
        if contourPerspective[index].size == 8:
            area = cv.contourArea(val)
            if (area > largest_area):
                largest_area = area
                largestContourIndex = index
                print("Found a bigger 4-sided contour index: ", largestContourIndex, " Vals: ",
                      contourPerspective[index])

    # Draw the contours over the source image. The rectangle should be blue and all other contours green.

    #   cv.namedWindow("contours on threshold", cv.WINDOW_NORMAL)
    #   cv.drawContours(img, contourPerspective, -1, (0,255,0), 10)
    #   cv.drawContours(img, contourPerspective, largestContourIndex, (255,0,0), 10)
    #   cv.imshow("contours on threshold", cv.resize(blur, (0,0), fx=scalefactor, fy=scalefactor))

    # Apply the four-point transform from https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
    # This broadly works, but needs more. It needs to not crop out the left side of the image, or the top and bottom.
    # This should also return a DPI value for use with the DXF conversion step to insure consistent scale.

    # What this actually needs to be is a checkerboard calibration.
    warped = four_point_transform(img, contourPerspective[largestContourIndex].reshape(4, 2))

    return warped

import cv2 as cv
from unwarp import four_point_transform

src = cv.imread("sampleImages/4.jpg", 1)

#scale factor for image displays only
scalefactor = 0.1

#display the source image
cv.imshow("Source", cv.resize(src, (0,0), fx=scalefactor, fy=scalefactor))

#apply a meanshift and gaussian blur to smooth things out. This takes WAY too long
meanshift = cv.pyrMeanShiftFiltering(src, 20, 45,0 )
blur = cv.GaussianBlur(meanshift, (5,5), 2)
cv.imshow("Source Meanshift Blur", cv.resize(blur, (0,0), fx=scalefactor, fy=scalefactor))

#convert the image to grayscale
src_gray = cv.cvtColor(blur, cv.COLOR_BGR2GRAY)
cv.imshow("Source Gray", cv.resize(src_gray, (0,0), fx=scalefactor, fy=scalefactor))

#apply an inverse binary otsu perspective to turn it black and white, and hopefully split the histogram right down the middle
retval, ThresholdPerspective = cv.threshold(src_gray, 127, 255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
cv.imshow("Source Thresh", cv.resize(ThresholdPerspective, (0,0), fx=scalefactor, fy=scalefactor))

#find all the contours in the image
img, contourPerspective, hierarchyPerspective = cv.findContours(ThresholdPerspective,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)

# Find the largest rectangular contour in the image. This should error out if it doesn't find one
# the rectangular contour is a ruler, 6" x 1" in the sample images. It's needed for scaling and deskewing

largestArea = 0
largestContourIndex = 0
for index, val in enumerate(contourPerspective):
    contourPerspective[index] = cv.approxPolyDP(val,13,True)
    if(contourPerspective[index] .size == 8):
        area = cv.contourArea(val)
        if (area > largestArea):
            largestArea = area
            largestContourIndex = index
            print("Found a bigger 4-sided contour index: ",largestContourIndex, " Vals: ",contourPerspective[index])

#Draw the contours over the source image. The rectangle should be blue and all other contours green.

cv.namedWindow("contours on threshold", cv.WINDOW_NORMAL)
cv.drawContours(blur, contourPerspective, -1, (0,255,0), 10)
cv.drawContours(blur, contourPerspective, largestContourIndex, (255,0,0), 10)
cv.imshow("contours on threshold", cv.resize(blur, (0,0), fx=scalefactor, fy=scalefactor))

#Apply the four-point transform from https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
#This broadly works, but needs more. It needs to not crop out the left side of the image, or the top and bottom.
#This should also return a DPI value for use with the DXF conversion step to insure consistent scale.
warped = four_point_transform(blur, contourPerspective[largestContourIndex].reshape(4,2))
cv.imshow("warped", cv.resize(warped, (0,0), fx=scalefactor, fy=scalefactor))

cv.waitKey(0)
cv.destroyAllWindows()






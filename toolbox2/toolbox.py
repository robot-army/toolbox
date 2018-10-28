import cv2 as cv
from fixperspective import fixperspective

src = cv.imread("sampleImages/4.jpg", 1)

#scale factor for image displays only
scalefactor = 0.1

#display the source image
cv.imshow("Source", cv.resize(src, (0,0), fx=scalefactor, fy=scalefactor))

#apply a meanshift and gaussian blur to smooth things out. This takes WAY too long
#meanshift = cv.pyrMeanShiftFiltering(src, 20, 45,0 )
blur = cv.GaussianBlur(src, (5,5), 2)
cv.imshow("Source Meanshift Blur", cv.resize(blur, (0,0), fx=scalefactor, fy=scalefactor))

#convert the image to grayscale
src_gray = cv.cvtColor(blur, cv.COLOR_BGR2GRAY)
cv.imshow("Source Gray", cv.resize(src_gray, (0,0), fx=scalefactor, fy=scalefactor))

#apply an inverse binary otsu perspective to turn it black and white, and hopefully split the histogram right down the middle
retval, ThresholdPerspective = cv.threshold(src_gray, 127, 255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
cv.imshow("Source Thresh", cv.resize(ThresholdPerspective, (0,0), fx=scalefactor, fy=scalefactor))

warped = fixperspective(ThresholdPerspective)

cv.imshow("warped", cv.resize(warped, (0,0), fx=scalefactor, fy=scalefactor))

cv.waitKey(0)
cv.destroyAllWindows()






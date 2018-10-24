import cv2 as cv
import numpy as np

src = cv.imread("test.jpg", 1)
cv.imshow("Source", src)

src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
cv.imshow("Source Gray", src_gray)

retval, ThresholdPerspective = cv.threshold(src_gray, 70, 255, cv.THRESH_BINARY_INV)

img, contourPerspective, hierarchyPerspective = cv.findContours(ThresholdPerspective,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)

# print(contourPerspective)
# print("Starting...")

# largestArea = 0
# largestContourIndex = -1

for index, val in enumerate(contourPerspective):
	contourPerspective[index] = cv.approxPolyDP(val,10,True)
	# print(index)
	# print(val.size)

	if(val.size==4):
		print(index)
# 		print(val)
# 		area = cv.contourArea(val)
# 		print(area)
# 		if (area > largestArea):
# 			largestArea = area
# 			largestContourIndex = index

# print(largestContourIndex)
cv.namedWindow("contour Gray", cv.WINDOW_NORMAL)
cv.drawContours(src_gray, contourPerspective, -1, 3)
cv.imshow("contour Gray", img)


cv.waitKey(0)
cv.destroyAllWindows()

import cv2 as cv
from fixperspective import fixperspective
from preprocess import preprocess
from orientation import orientation

src = cv.imread("checkerboard\IMG_20181101_202653.jpg", 1)

#scale factor for image displays only
scalefactor = 0.1

cv.imshow("Source", cv.resize(src, (0, 0), fx=scalefactor, fy=scalefactor))

#orientation(cv.resize(src, (0, 0), fx=1, fy=1))
preproc = preprocess(src)

warped = fixperspective(preproc)

cv.imshow("Warped", cv.resize(warped, (0, 0), fx=scalefactor, fy=scalefactor))

cv.waitKey(0)
cv.destroyAllWindows()

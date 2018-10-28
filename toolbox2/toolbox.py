import cv2 as cv
from fixperspective import fixperspective
from preprocess import preprocess

src = cv.imread("sampleImages/4.jpg", 1)

# scale factor for image displays only
scalefactor = 0.1

# display the source image
cv.imshow("Source", cv.resize(src, (0, 0), fx=scalefactor, fy=scalefactor))

preproc = preprocess(src)

warped = fixperspective(preproc)

cv.imshow("Warped", cv.resize(warped, (0, 0), fx=scalefactor, fy=scalefactor))

cv.waitKey(0)
cv.destroyAllWindows()

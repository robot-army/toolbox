import numpy as np
import cv2
import glob
from unwarp import four_point_transform
import pickle
from calibration import generatecalibration
from preprocess import preprocess
from orientation import orientation

try:
    with open('checkerboard/calibration.pickle', 'rb') as f:
        imagesInput,ret,mtx,dist,rvecs,tvecs = pickle.load(f)
        print("Existing calibration data found")
except:
    print("Error reading file, calibrating")
    imagesInput = []
    ret = 0
    mtx = 0
    dist = 0
    rvecs = 0
    tvecs = 0

images = glob.glob('checkerboard/*.jpg')

if images != imagesInput:
    print("Input names are different, calibrating")
    ret, mtx, dist, rvecs, tvecs = generatecalibration(images)
else:
    images = imagesInput

#these need to be the same as in the calibration one...they should be able to be independent, but funny things happen
scalefactor_processing = 0.5
scalefactor_display = 0.5   #This is on top of the processing one

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

imgBIG2 = cv2.imread('checkerboard/IMG_20181101_202640.jpg')
img = cv2.resize(imgBIG2, (0, 0), fx=scalefactor_processing, fy=scalefactor_processing)
cv2.imshow("testasdf",cv2.resize(img, (0, 0), fx=scalefactor_display, fy=scalefactor_display))

h,  w = img.shape[:2]
newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))


cv2.imshow("input",cv2.resize(img, (0, 0), fx=scalefactor_display, fy=scalefactor_display))

# undistort
print("Undistorting")
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
print("mtx",mtx,"dist",dist,"newcameramtx",newcameramtx)
print("grayscaling")
gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)

ret, corners = cv2.findChessboardCorners(gray, (4, 4), None)
corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
outercorners = np.array((corners2[0], corners2[3], corners2[12], corners2[15]))
print("outer corners: ",outercorners)

unwarp = four_point_transform(dst, outercorners.reshape(4, 2))

cv2.imshow("unwarp", cv2.resize(unwarp, (0, 0), fx=scalefactor_display, fy=scalefactor_display))

cv2.imshow("undistort",cv2.resize(dst, (0, 0), fx=scalefactor_display, fy=scalefactor_display))

cv2.circle(unwarp,(500,500),100,(0,255,0),10)
cv2.rectangle(unwarp,(500-300,500-150),(500+396+180 ,500+396+150),(200,200,200),-1)

cv2.imshow("fill",cv2.resize(unwarp, (0, 0), fx=scalefactor_display, fy=scalefactor_display))

unwarp[np.where((unwarp==[0,0,0]).all(axis=2))] = [200,200,200]
cv2.imshow("blacktowhite",cv2.resize(unwarp, (0, 0), fx=scalefactor_display, fy=scalefactor_display))

preproc = preprocess(unwarp)
cv2.imshow("preproc",cv2.resize(preproc, (0, 0), fx=scalefactor_display, fy=scalefactor_display))

#TODO - Make this thing crop to contours.
#TODO - Fix the structure/refactor it so this checkerboard is default.
#TODO - Fix the processing so it doesn't exclude small contours...this won't work for things with springs in them

orientation(preproc)
cv2.waitKey(0)
cv2.destroyAllWindows()
import numpy as np
import cv2
import glob
from unwarp import four_point_transform
import pickle
import time
import multiprocessing


# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((4*4,3), np.float32)
objp[:,:2] = np.mgrid[0:4,0:4].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob('checkerboard/*.jpg')

scalefactor = 0.3

start = time.time()

jobs = []

for fname in images:
    print("Name: ",fname)
    img = cv2.imread(fname,1)
    img = cv2.resize(img, (0, 0), fx=scalefactor, fy=scalefactor)

    cv2.imshow("cbd",img)

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #    cv2.imshow("gray",gray)
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (4,4),None)

    # If found, add object points, image points (after refining them)
    if ret == True:
#        print("Found points", corners)
        objpoints.append(objp)
        #
        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)
        #
        #     # Draw and display the corners
#        img = cv2.drawChessboardCorners(gray, (4,4), corners2,ret)
#        outercorners = np.array((corners2[0],corners2[3],corners2[12],corners2[15]))
#        print("corners",outercorners.reshape(4,2))
#        unwarp = four_point_transform(img,outercorners.reshape(4,2))
#        cv2.imshow("cbd", unwarp)
#        cv2.waitKey(500)

print(time.time()-start)


ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

calibdata = images,ret,mtx,dist,rvecs,tvecs


print("Pickling: ",calibdata)
with open('checkerboard/calibration.pickle', 'wb') as f:
    images2 = pickle.dump(calibdata,f,pickle.HIGHEST_PROTOCOL)


images = 0
ret = 0
mtx = 0
dist = 0
rvecs = 0
tvecs = 0


with open('checkerboard/calibration.pickle', 'rb') as f:
    images,ret,mtx,dist,rvecs,tvecs = pickle.load(f)




imgBIG2 = cv2.imread('checkerboard/IMG_20181101_202640.jpg')
img = cv2.resize(imgBIG2, (0, 0), fx=scalefactor, fy=scalefactor)
cv2.imshow("testasdf",img)

h,  w = img.shape[:2]
newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))


cv2.imshow("input",img)

# undistort
print("Undistorting")
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
print("mtx",mtx,"dist",dist,"newcameramtx",newcameramtx)
print("grayscaling")
gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)

ret, corners = cv2.findChessboardCorners(gray, (4, 4), None)
corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
outercorners = np.array((corners2[0], corners2[3], corners2[12], corners2[15]))

unwarp = four_point_transform(dst, outercorners.reshape(4, 2))

cv2.imshow("unwarp", unwarp)

cv2.imshow("undistort",dst)

cv2.waitKey(0)
cv2.destroyAllWindows()
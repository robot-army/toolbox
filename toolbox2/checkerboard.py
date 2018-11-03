import numpy as np
import cv2
import glob
from unwarp import four_point_transform
import pickle
import time
import multiprocessing
from queue import Queue
from threading import Thread


print("multiprocessing says", multiprocessing.cpu_count())


# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((4*4,3), np.float32)
objp[:,:2] = np.mgrid[0:4,0:4].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.


images = glob.glob('checkerboard/*.jpg')

scalefactor_processing = 0.5
scalefactor_display = 0.5    #This is on top of the processing one

start = time.time()

jobs = []


def worker():
    while True:

#        img = cv2.UMat(q.get())
        img = q.get()
        print("Starting")
        img = cv2.resize(img, (0, 0), fx=scalefactor_processing, fy=scalefactor_processing)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (4, 4), None)
# If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
 #          corners2 = cv2.UMat.get(cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria))
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
#        print("Finishing - objp",objp)
        q.task_done()

q = Queue()
num_worker_threads = 2
#on macbook, with 0.5 scaledown and 2 cores -
# 9.1 s with 2 threads
# 14 s with 1 thread
# 9.75 s with 4 threads
# 9.1 s with 3 threads
#

for i in range(num_worker_threads):
    t = Thread(target=worker)
    t.daemon = True
    t.start()

img = cv2.imread('checkerboard/IMG_20181101_202640.jpg')
img = cv2.resize(img, (0, 0), fx=scalefactor_processing, fy=scalefactor_processing)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

for fname in images:
    img = cv2.imread(fname, 1)
    q.put(img)
q.join()

print(time.time()-start)

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

calibdata = images,ret,mtx,dist,rvecs,tvecs

#print("Pickling: ",calibdata)
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

unwarp = four_point_transform(dst, outercorners.reshape(4, 2))

cv2.imshow("unwarp", cv2.resize(unwarp, (0, 0), fx=scalefactor_display, fy=scalefactor_display))

cv2.imshow("undistort",cv2.resize(dst, (0, 0), fx=scalefactor_display, fy=scalefactor_display))

cv2.waitKey(0)
cv2.destroyAllWindows()
import numpy as np
import cv2
import glob
from unwarp import four_point_transform
import pickle
from calibration import generatecalibration
from preprocess import preprocess
from orientation import orientation




forcecalibration = False


def mse(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])

    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err

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


if (images != imagesInput) or forcecalibration:
    print("Input names are different, calibrating")
    ret, mtx, dist, rvecs, tvecs = generatecalibration(images)
else:
    images = imagesInput

images = glob.glob('checkerboard/*.jpg')

for fname in images:
    _, imagename = fname.split('/')

    errors = 0
    #these need to be the same as in the calibration one...they should be able to be independent, but funny things happen
    scalefactor_processing = 0.5
    scalefactor_display = 0.5   #This is on top of the processing one

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    imgBIG2 = cv2.imread(fname)
    img = cv2.resize(imgBIG2, (0, 0), fx=scalefactor_processing, fy=scalefactor_processing)

    manualname = "manual_thresh/"+imagename
   # print(manualname)
    manual_thresh = cv2.imread(manualname)

    cv2.imshow("Manualthresh",manual_thresh)

    manual_thresh = cv2.cvtColor(manual_thresh, cv2.COLOR_BGR2GRAY)
    retval, manual_thresh = cv2.threshold(manual_thresh, 50, 255, cv2.THRESH_BINARY_INV)

    gray2, contoursmanual, hierarchymanual = cv2.findContours(manual_thresh, cv2.RETR_EXTERNAL,
                                                                       cv2.CHAIN_APPROX_NONE)
    contourmanual = contoursmanual[0]

    gray3 = cv2.cvtColor(gray2, cv2.COLOR_GRAY2BGR)

    cv2.drawContours(gray3, contoursmanual, -1, (0, 255, 0), 2)
    cv2.imshow("contours on manual threshold", gray3)

    #cv2.imshow("testasdf",cv2.resize(img, (0, 0), fx=scalefactor_display, fy=scalefactor_display))

    h,  w = img.shape[:2]
    newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))


    cv2.imshow("input",cv2.resize(img, (0, 0), fx=scalefactor_display, fy=scalefactor_display))

    # undistort
  #  print("Undistorting")
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
  #  print("mtx",mtx,"dist",dist,"newcameramtx",newcameramtx)
  #  print("grayscaling")
    gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, (4, 4), None)
    try:
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    except:
        print("couldn't get corner subpix")
        errors = 1
    outercorners = np.array((corners2[0], corners2[3], corners2[12], corners2[15]))
  #  print("outer corners: ",outercorners)

    unwarp = four_point_transform(dst, outercorners.reshape(4, 2))

    #cv2.imshow("unwarp", cv2.resize(unwarp, (0, 0), fx=scalefactor_display, fy=scalefactor_display))

    #cv2.imshow("undistort",cv2.resize(dst, (0, 0), fx=scalefactor_display, fy=scalefactor_display))

    cv2.rectangle(unwarp,(500-300,500-150),(500+396+180 ,500+396+150),(200,200,200),-1)

    #cv2.imshow("fill",cv2.resize(unwarp, (0, 0), fx=scalefactor_display, fy=scalefactor_display))

    unwarp[np.where((unwarp==[0,0,0]).all(axis=2))] = [200,200,200]
    cv2.imshow("blacktowhite",cv2.resize(unwarp, (0, 0), fx=scalefactor_display, fy=scalefactor_display))

    gray = cv2.cvtColor(unwarp, cv2.COLOR_BGR2GRAY)
    retval, gray = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    #cv2.imshow("gray",gray)
    img, contourPerspective, hierarchyPerspective = cv2.findContours(gray, cv2.RETR_EXTERNAL,
                                                                    cv2.CHAIN_APPROX_SIMPLE)

    # rect = cv2.minAreaRect(cnt)
    # box = cv2.boxPoints(rect)
    # box = np.int0(box)
    #cv2.drawContours(img,[box],0,(0,0,255),2)

    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    largest_area = 0;

    for index, val in enumerate(contourPerspective):
        contourPerspective[index] = cv2.approxPolyDP(val, 2, True)
        area = cv2.contourArea(val)
        if area > 100:
            M = cv2.moments(val)
            x1 = int(M['m10'] / M['m00'])
            y1 = int(M['m01'] / M['m00'])
            width, height = img.shape[:2]
            x2 = width // 2
            y2 = height // 2
            distance = np.hypot(x2 - x1, y2 - y1)
       #     print("Drawing contour size", area )
       #     print("Distance ", distance)
            if (area > largest_area) and (distance < 700):
                largest_area = area
                largestContourIndex = index
            #    print("Found a bigger contour index: ", largestContourIndex, " Vals: ",contourPerspective[index])


    cv2.drawContours(img, contourPerspective, -1, (0,255,0), 10)
    cv2.drawContours(img, contourPerspective,largestContourIndex,(255,0,0),10)

    x,y,w,h = cv2.boundingRect(contourPerspective[largestContourIndex])
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cropped = unwarp[y-30:y+h+30, x-30:x+w+30]
    try:
        cv2.imshow("cropped", cropped)
    except:
        print("couldn't show cropped image")
        errors = 1

    rect = cv2.minAreaRect(contourPerspective[largestContourIndex])
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(img, [box], 0, (0, 0, 255), 2)

    cv2.namedWindow("contours on threshold", cv2.WINDOW_NORMAL)
    cv2.imshow("contours on threshold", img)

    if errors == 1:
        print("errors on ",fname)

 #   orientation(cropped)

    cropped = cv2.pyrMeanShiftFiltering(cropped, 20, 45,0 )


    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cropped, 50, 150, apertureSize=3)

    retval, gray2 = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)

    print("MSE: ",mse(gray2,manual_thresh))

    gray2, contourPerspective, hierarchyPerspective = cv2.findContours(gray2, cv2.RETR_EXTERNAL,
                                                                    cv2.CHAIN_APPROX_NONE)
    gray3 = cv2.cvtColor(gray2, cv2.COLOR_GRAY2BGR)

    for contourautomatic in contourPerspective:
        ret = cv2.matchShapes(contourmanual, contourautomatic, 1, 0.0)
       # print("Contourmatching: ",ret)
        cv2.drawContours(gray3, contourautomatic, -1, (0,255,0), 2)
        cv2.imshow("contours on cropped threshold", gray3)
        #cv2.waitKey(0)


    cv2.imshow("Canny",edges)
    minLineLength = 100
    maxLineGap = 10
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength, maxLineGap)
    try:
        for i in range(1,len(lines)):
            for x1, y1, x2, y2 in lines[i]:
     #           print("vals",x1,y1,x2,y2)
                cv2.line(cropped, (x1, y1), (x2, y2), (0, 255, 0), 2)
    except:
        print("Couldn't find any lines to draw")

    cv2.imshow("lines",cropped)

    print("processing: ",fname)
 #   cv2.waitKey(500)

    #TODO - Check dimensions of final images
    #TODO - Parallelize the worker doing the processing
    #TODO - Use both the lines and the PCA to find orientation
    #TODO - Fix the structure/refactor it so this checkerboard is default.
    #TODO - Fix the processing so it doesn't exclude small contours...this won't work for things with springs in them


print("Waiting for keypress")
cv2.waitKey(0)
cv2.destroyAllWindows()
import numpy as np
import cv2
import glob
from unwarp import four_point_transform
import pickle
from calibration import generatecalibration

forcecalibration = False

np.set_printoptions(precision=2)



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

for fname in images:

    print("processing: ",fname)

    _, imagename = fname.split('/')

    # Read in manually thresholded image and find contours on it

    manualname = "manual_thresh/" + imagename
    manual_thresh = cv2.imread(manualname)
    cv2.imshow("Manualthresh", manual_thresh)
    manual_thresh = cv2.cvtColor(manual_thresh, cv2.COLOR_BGR2GRAY)
    retval, manual_thresh = cv2.threshold(manual_thresh, 50, 255, cv2.THRESH_BINARY_INV)
    gray2, contoursmanual, hierarchymanual = cv2.findContours(manual_thresh, cv2.RETR_EXTERNAL,
                                                              cv2.CHAIN_APPROX_NONE)
    gray3 = cv2.cvtColor(gray2, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(gray3, contoursmanual, -1, (0, 255, 0), 2)
    cv2.imshow("contours on manual threshold", gray3)

    errors = 0

    # Set up some constants, for general processing and checkerboard detection

    #these need to be the same as in the calibration one...they should be able to be independent, but funny things happen
    scalefactor_processing = 0.5
    scalefactor_display = 0.5   #This is on top of the processing one

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Read in raw image

    imgBIG2 = cv2.imread(fname)
    img = cv2.resize(imgBIG2, (0, 0), fx=scalefactor_processing, fy=scalefactor_processing)

    h,  w = img.shape[:2]
    newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

    cv2.imshow("input",cv2.resize(img, (0, 0), fx=scalefactor_display, fy=scalefactor_display))

    # undistort input image - checkerboard, then convert to grayscale
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)

    # Find chessboard corners in grayscale image
    ret, corners = cv2.findChessboardCorners(gray, (4, 4), None)
    try:
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    except:
        print("couldn't get corner subpix")
        errors = 1

    # Use the four outside chessboard corners to unwarp the image ('birds eye view', constant pixels/mm)
    outercorners = np.array((corners2[0], corners2[3], corners2[12], corners2[15]))
    unwarp = four_point_transform(dst, outercorners.reshape(4, 2))

    # Draw a rectangle of some random close to background colour over the chessboard to hide it
    cv2.rectangle(unwarp,(500-300,500-150),(500+396+180 ,500+396+150),(200,200,200),-1)

    # use some funky-ass python stuff from off t'internet to turn all the black pixels to the same bg colour
    unwarp[np.where((unwarp==[0,0,0]).all(axis=2))] = [200,200,200]

    # Find all the contours on the unwarped, de-checkerboarded, de-backgrounded image
    gray = cv2.cvtColor(unwarp, cv2.COLOR_BGR2GRAY)
    retval, gray = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    img, contourPerspective, hierarchyPerspective = cv2.findContours(gray, cv2.RETR_EXTERNAL,
                                                                    cv2.CHAIN_APPROX_SIMPLE)

    # Make somewhere to draw those contours
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Find the biggest contour near enough to the centre of the image so we can crop to it in the next step
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
            if (area > largest_area) and (distance < 700):
                largest_area = area
                largestContourIndex = index

    # Get the bounding rectangle of the largest contour and crop to it
    x,y,w,h = cv2.boundingRect(contourPerspective[largestContourIndex])
    cropped = unwarp[y-30:y+h+30, x-30:x+w+30]
    try:
        cv2.imshow("cropped", cropped)
    except:
        print("couldn't show cropped image")
        errors = 1

    # Routine for drawing the cropping area - For debugging
    # cv2.drawContours(img, contourPerspective, -1, (0,255,0), 10)
    # cv2.drawContours(img, contourPerspective,largestContourIndex,(255,0,0),10)
    # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # rect = cv2.minAreaRect(contourPerspective[largestContourIndex])
    # box = cv2.boxPoints(rect)
    # box = np.int0(box)
    # cv2.drawContours(img, [box], 0, (0, 0, 255), 2)
    #
    # cv2.namedWindow("contours on threshold", cv2.WINDOW_NORMAL)
    # cv2.imshow("contours on threshold", img)

    if errors == 1:
        print("errors on ",fname)

    cropped = cv2.pyrMeanShiftFiltering(cropped, 20, 45,0 )
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cropped, 50, 150, apertureSize=3)
    retval, gray2 = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    gray2, contourPerspective, hierarchyPerspective = cv2.findContours(gray2, cv2.RETR_EXTERNAL,
                                                                    cv2.CHAIN_APPROX_NONE)
    gray3 = cv2.cvtColor(gray2, cv2.COLOR_GRAY2BGR)

    fail = 0

    for contourautomatic in contourPerspective:
        for contourmanual in contoursmanual:
            fail = 0
            ret = cv2.matchShapes(contourmanual, contourautomatic, 1, 0.0)
            try:
                areamatch = cv2.contourArea(contourmanual) / cv2.contourArea(contourautomatic)
            except:
                pass
            if (ret != 1.7976931348623157e+308) and (ret != 0) and (0.5 < areamatch) and (areamatch < 2):
                print("Contourmatching: ",ret, "AreaMatch: ",areamatch, "MSE: ",mse(gray2,manual_thresh))
                cv2.drawContours(gray3,contourautomatic,-1,(255,0,0),3)
                fail = 0
            else:
                fail = 1
            cv2.drawContours(gray3, contourautomatic, -1, (0,255,0), 2)
            cv2.imshow("contours on cropped threshold", gray3)
#            cv2.waitKey(0)
    if fail == 1:
        print("No matching contours")

    #TODO - Check dimensions of final images
    #TODO - Parallelize the worker doing the processing
    #TODO - Use both the lines and the PCA to find orientation
    #TODO - Fix the structure/refactor it so this checkerboard is default.
    #TODO - Fix the processing so it doesn't exclude small contours...this won't work for things with springs in them

print("Waiting for keypress")
cv2.waitKey(0)
cv2.destroyAllWindows()
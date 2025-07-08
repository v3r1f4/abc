import cv2 as cv
import cv2.aruco as aruco
import glob

gray = glob.glob('output/10_preprocessed.jpg')

dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_1000)
parameters = aruco.DetectorParameters()
detector = aruco.ArucoDetector(dictionary, parameters)

for fname in gray:
    img = cv.imread(fname)
    markerCorners, markerIds, rejectedImgPoints = detector.detectMarkers(img)

    output = aruco.drawDetectedMarkers(img.copy(), markerCorners, markerIds)
    cv.imshow('Detected markers', output)
    cv.waitKey(0)
    cv.destroyAllWindows()
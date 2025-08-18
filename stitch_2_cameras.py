import cv2 as cv
import imutils
import numpy as np

def select_descriptor_method(image, method=None):
    assert method is not None, " 'sift', 'surf', 'orb', 'brisk' "
     
    if method == 'sift':
        descriptor = cv.SIFT.create()
    elif method == 'surf':
        descriptor = cv.xfeatures2d.SURF.create()
    elif method == 'orb':
        descriptor = cv.ORB.create()
    elif method == 'brisk':
        descriptor = cv.BRISK.create()

    keypoints, features = descriptor.detectAndCompute(image, None)

    return keypoints, features

def main():
    cam1 = cv.VideoCapture(2, cv.CAP_DSHOW)
    cam1.set(cv.CAP_PROP_FRAME_WIDTH, 1200)
    cam1.set(cv.CAP_PROP_FRAME_HEIGHT, 800)

    cam2 = cv.VideoCapture(1, cv.CAP_DSHOW)
    cam2.set(cv.CAP_PROP_FRAME_WIDTH, 1200)
    cam2.set(cv.CAP_PROP_FRAME_HEIGHT, 800)

    while 1:
        _, frame1 = cam1.read()
        _, frame2 = cam2.read()

        frame1 = cv.cvtColor(frame1, cv.COLOR_BGR2RGB)
        frame1_gray = cv.cvtColor(frame1, cv.COLOR_RBG2GRAY)

        frame2 = cv.cvtColor(frame2, cv.COLOR_BGR2RGB)
        frame2_gray = cv.cvtColor(frame2, cv.COLOR_RBG2GRAY)

        keypoints_frame1,

        cv.imshow('frame 1', frame1)
        cv.imshow('frame 2', frame2)

        if cv.waitKey(1) == ord('q'):
                break
        
    cam1.release()
    cam2.release()
    cv.destroyAllWindows()
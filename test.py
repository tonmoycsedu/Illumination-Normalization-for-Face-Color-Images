# import the necessary packages
from imutils import face_utils
import numpy as np
import imutils
import dlib
import cv2
 


# the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("./Hog_face_detector/shape_predictor_68_face_landmarks.dat")
 
# load the input image, resize it, and convert it to grayscale
image = cv2.imread("Img37.jpg")
image = imutils.resize(image, width=500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
 
# detect faces in the grayscale image
rects = detector(gray, 1)
# loop over the face detections
for (i, rect) in enumerate(rects):
    shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)
 
    # convert dlib's rectangle to a OpenCV-style bounding box
    # [i.e., (x, y, w, h)], then draw the face bounding box
    #(x, y, w, h) = face_utils.rect_to_bb(rect)
    #cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
 
    # show the face number
    #cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
     #   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
 
    # loop over the (x, y)-coordinates for the facial landmarks
    # and draw them on the image
    for (x, y) in shape:
        #print(x,y)
        cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
        
     
    for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
        print(name)
        count = 0
        #print(shape[i:j])
        if(name == "jaw"):
            k=0
            while(k<9):
                x1 = shape[i:j][k][0]
                y1 = shape[i:j][k][1]
                nextColumnY = shape[i:j][k+1][1]
                x2 = shape[i:j][16-k][0]
                print(x1,y1,x2,y1,nextColumnY)
                while(y1<nextColumnY):
                    while(x1<x2):
                        count = count+1
                        cv2.circle(image, (x1, y1), 1, (0, 210, 255), -1) 
                        x1 = x1+10
                        
                    y1 = y1+10
                    x1 = shape[i:j][k][0]

                k=k+1
    print(count)

# show the output image with the face detections + facial landmarks
cv2.imshow("Output", image)
cv2.waitKey(0)                   
                   
                   
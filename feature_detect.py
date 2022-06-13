import cv2 
import numpy as np

cap =cv2.VideoCapture(0) #webcam

orb =cv2.ORB_create()

#image
img = cv2.imread('cube5.jpg',0)
img= cv2.resize(img,(400,400),interpolation=cv2.INTER_AREA)
#img = np.ones((600,600,3), np.uint8)  * 255 
#cv2.line(img,(50,200),(550,200),(0,0,0),6)  
#cv2.line(img,(50,400),(550,400),(0,0,0),6)
#cv2.line(img,(200,50),(200,550),(0,0,0),6)
#cv2.line(img,(400,50),(400,550),(0,0,0),6)
#cv2.imshow('img',img)       
kp1, des1 =orb.detectAndCompute(img,None)
img = cv2.drawKeypoints(img,kp1,None)

while(1):
    ret, frame =cap.read() 

    #frame
    kp2, des2 =orb.detectAndCompute(frame,None)
    frame = cv2.drawKeypoints(frame,kp2,None)

    #matching
    bf = cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck= True)
    matches = bf.match(des1,des2)
    matches = sorted(matches, key = lambda x:x.distance)
    matching_result = cv2.drawMatches(img,kp1,frame,kp2,matches[:10],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    cv2.imshow('result',matching_result)
    #cv2.imshow('frame',frame)
    #cv2.imshow('image',img)

    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break   
cap.release()
cv2.destoryAllwindows()
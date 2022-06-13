import cv2 #opencv函式庫
import numpy as np #numpy函式庫
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import os

c=cv2.VideoCapture(0) #攝像頭輸入

c.set(3,640)
c.set(4,480)
segmentor = SelfiSegmentation()
fpsReader = cvzone.FPS()

while(1):
    ret, frame =c.read() 
    lower_red = np.array([165,50,30]) #設置上下界
    upper_red = np.array([175,255,255])
    if ret:
        #a=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV) #用cvtColor把彩色轉為HSV
        #red_mask = cv2.inRange(a, lower_red, upper_red) #設置遮罩
        #red_only = cv2.bitwise_and(frame,frame, mask=red_mask) #過濾完後會是黑白的 把白色轉為要過濾的顏色
        frame1=segmentor.removeBG(frame,(255,0,255),0.8)
        imgstack = cvzone.stackImages([frame,frame1],2,1)
        cv2.imshow('frame',imgstack)
        #cv2.imshow('mask',red_mask) 
        #cv2.imshow('red_only',red_only)
        if cv2.waitKey(1) & 0xFF == ord('q'): #waitKey為opencv內建函式,等待輸入 #0xFF為了防止BUG 
            break   
    else:
        break
c.release()
cv2.destoryAllwindows()

import cv2 
import numpy as np 

def no(x):
    pass

c=cv2.VideoCapture(0)
cv2.namedWindow("Track")
cv2.createTrackbar('L H','Track',0,179,no)
cv2.createTrackbar('L S','Track',0,255,no)
cv2.createTrackbar('L V','Track',0,255,no)
cv2.createTrackbar('U H','Track',0,179,no)
cv2.createTrackbar('U S','Track',0,255,no)
cv2.createTrackbar('U V','Track',0,255,no)
while(1):
    ret, frame =c.read() 
   #lower_red = np.array([165,50,30]) 
   #upper_red = np.array([175,255,255])
    lh = cv2.getTrackbarPos('L H','Track')
    ls = cv2.getTrackbarPos('L S','Track')
    lv = cv2.getTrackbarPos('L V','Track')
    uh = cv2.getTrackbarPos('U H','Track')
    us = cv2.getTrackbarPos('U S','Track')
    uv = cv2.getTrackbarPos('U V','Track')
    lower_red = np.array([lh,ls,lv]) 
    upper_red = np.array([uh,us,uv])
    if ret:
        a=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV) #用cvtColor把彩色轉為HSV
        red_mask = cv2.inRange(a, lower_red, upper_red) #設置遮罩
        red_only = cv2.bitwise_and(frame,frame, mask=red_mask) #過濾完後會是黑白的 把白色轉為要過濾的顏色
        #cv2.imshow('frame',frame)
        #cv2.imshow('mask',red_mask) 
        cv2.imshow('red_only',red_mask)
        if cv2.waitKey(1) & 0xFF == ord('q'): #waitKey為opencv內建函式,等待輸入 #0xFF為了防止BUG 
            break   
    else:
        break
c.release()
cv2.destoryAllwindows()

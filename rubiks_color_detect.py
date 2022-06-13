import cv2 
import numpy as np
cap =cv2.VideoCapture(0) #webcam

d=[]
q={}
def detect_color(contours,color):
        for c in contours:
            x,y,w,h=cv2.boundingRect(c)
            area = cv2.contourArea(c)
            ep = 0.005 * cv2.arcLength(c,True)  #精度
            approx = cv2.approxPolyDP(c, ep, True)
            if area>100 and w/h >0.9 and w/h <1.1 and len(approx)<40:
                s=tuple(str(c)) #list不能當key
                q.update({s: color})   #儲存顏色資訊
                d.append(c)

def x_contour(c):
    x,y,w,h=cv2.boundingRect(c)  
    return x  

def y_contour(c):
    x,y,w,h=cv2.boundingRect(c)  
    return y 

def track(a):
    x,y,w,h=cv2.boundingRect(a)
    u = int(x+0.2*w)
    v = int(y+0.2*h)
    a = int(x+0.8*w)
    b = int(y+0.8*h)
    cv2.rectangle(frame,(u,v),(a,b),(0,0,0),2)

#  1 4 7    
#  2 5 8
#  3 6 9
#畫格子
def draw_rectangle(frame):  
    cv2.rectangle(frame,(29,29),(56,56),(0,0,0),3)     #1
    cv2.rectangle(frame,(29,70),(56,97),(0,0,0),3)     #2
    cv2.rectangle(frame,(29,111),(56,138),(0,0,0),3)   #3
    cv2.rectangle(frame,(69,29),(96,56),(0,0,0),3)     #4
    cv2.rectangle(frame,(69,70),(96,97),(0,0,0),3)     #5
    cv2.rectangle(frame,(69,111),(96,138),(0,0,0),3)   #6
    cv2.rectangle(frame,(110,29),(137,56),(0,0,0),3)   #7
    cv2.rectangle(frame,(110,70),(137,97),(0,0,0),3)   #8
    cv2.rectangle(frame,(110,111),(137,138),(0,0,0),3) #9

lower_red = np.array([165,190,100],np.uint8) #rgb範圍
upper_red = np.array([179,255,255],np.uint8)
lower_ora = np.array([0,190,100],np.uint8) 
upper_ora = np.array([15,255,255],np.uint8)
lower_yel = np.array([27,190,100],np.uint8) 
upper_yel = np.array([54,255,255],np.uint8)
lower_blu = np.array([91,165,100],np.uint8) 
upper_blu = np.array([110,255,255],np.uint8)
lower_gre = np.array([58,165,100],np.uint8) 
upper_gre = np.array([75,255,255],np.uint8)
lower_whi = np.array([0,0,130],np.uint8) 
upper_whi = np.array([179,45,255],np.uint8)

img = np.zeros((512,512,3), np.uint8)
cv2.rectangle(img,(110,111),(137,138),(0,0,0),3) 

while(1):
    q={}
    d=[]
    ret, frame =cap.read() 
    a=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    r_mask = cv2.inRange(a, lower_red, upper_red) #遮罩
    o_mask = cv2.inRange(a, lower_ora, upper_ora)
    y_mask = cv2.inRange(a, lower_yel, upper_yel)
    b_mask = cv2.inRange(a, lower_blu, upper_blu)
    g_mask = cv2.inRange(a, lower_gre, upper_gre)
    w_mask = cv2.inRange(a, lower_whi, upper_whi)
    contours_r, _=cv2.findContours(r_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) #輪廓
    contours_o, _=cv2.findContours(o_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    contours_y, _=cv2.findContours(y_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    contours_b, _=cv2.findContours(b_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    contours_g, _=cv2.findContours(g_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    contours_w, _=cv2.findContours(w_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    #顯示的顏色
    detect_color(contours_r,(52,18,183))
    detect_color(contours_o,(0, 88, 255))
    detect_color(contours_y,(0,213,255))
    detect_color(contours_b,(173,70,0))
    detect_color(contours_g,(72,155,9))
    detect_color(contours_w,(255,255,255))

    d=sorted(d,key=x_contour)
    row=[]
    for (i,e) in enumerate(d, 1):
        row.append(e)
        if i==3 :
            row=sorted(row,key=y_contour)
            for (r,p) in enumerate(row, 1):
                y=tuple(str(p))
                if r==1:
                    cv2.rectangle(frame,(31,31),(54,54),q[y],-1)     #1
                    track(p)
                elif r==2:
                    cv2.rectangle(frame,(31,72),(54,95),q[y],-1)     #2
                    track(p)
                elif r==3:
                    cv2.rectangle(frame,(31,113),(54,136),q[y],-1)   #3
                    track(p)
            row=[]
        elif i==6:
            row=sorted(row,key=y_contour)
            for (r,p) in enumerate(row, 1):
                y=tuple(str(p))
                if r==1:
                    cv2.rectangle(frame,(71,31),(94,54),q[y],-1)     #4
                    track(p)
                elif r==2:
                    cv2.rectangle(frame,(71,72),(94,95),q[y],-1)     #5
                    track(p)
                elif r==3:
                    cv2.rectangle(frame,(71,113),(94,136),q[y],-1)   #6
                    track(p)
            row=[]
        elif i==9:
            row=sorted(row,key=y_contour)
            for (r,p) in enumerate(row, 1):
                y=tuple(str(p))
                if r==1:
                    cv2.rectangle(frame,(112,31),(135,54),q[y],-1)   #7
                    track(p)
                elif r==2:
                    cv2.rectangle(frame,(112,72),(135,95),q[y],-1)   #8
                    track(p)
                elif r==3:
                    cv2.rectangle(frame,(112,113),(135,136),q[y],-1) #9
                    track(p)
            row=[]
        
    draw_rectangle(frame)
    cv2.imshow('frame',frame)
    cv2.imshow('color record',img)
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break   
cap.release()
cv2.destoryAllwindows()

#畫出所有輪廓
#cv2.drawContours(frame, contours, -1, (0, 255, 0), 4) 

#畫矩形
#x,y,w,h=cv2.boundingRect(c)
#cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

#利用閉運算連接canny斷掉的線條
#fix = cv2.morphologyEx(canny,cv2.MORPH_CLOSE,kernel=(3,3),iterations=3)

#嘗試篩選正方形
#   for c in contours:
#        x,y,w,h=cv2.boundingRect(c)
#        if w/h>=0.9 and w/h<=1.1 and cv2.contourArea(c)>65 and cv2.contourArea(c)<90:
#            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

#顯示邊長數量
#字型 font=cv2.FONT_HERSHEY_COMPLEX
#t=int(len(approx))
#cv2.putText(frame,"{}".format(t),(x,y),font,1,(0))

#找出輪廓的前置作業
    #gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) #顏色空間轉換
    #blur=cv2.GaussianBlur(gray,(7,7),0) #高斯模糊 
    #canny=cv2.Canny(r_mask,50,150) #邊緣偵測
    #kernel = np.ones((3,3),np.uint8)
    #dila=cv2.dilate(canny,kernel,1) #加粗邊緣

#舊版的篩選條件
    #for c in contours:
    #    x,y,w,h=cv2.boundingRect(c)
    #    if w/h >0.8 and w/h <1.2 : #正方形
    #        ep = 0.0005 * cv2.arcLength(c,True)  # 精度
    #        approx = cv2.approxPolyDP(c, ep, True) #近似多邊形
    #        area = cv2.contourArea(approx) #面積
    #        if  area>550 and area <5500 and len(approx)>7 and len(approx)<40 : #篩選
    #            #cv2.drawContours(frame, [approx], -1, (187,255,255), 2)
    #            u = int(x+0.1*w)
    #            v = int(y+0.1*h)
    #            a = int(x+0.9*w)
    #            b = int(y+0.9*h)
    #            cv2.rectangle(frame,(u,v),(a,b),(255,105,180),3)

    #按大小 不穩
    #e=sorted(d,key=cv2.contourArea,reverse=True)
    #if len(e)>=9:
    #    for i in range(0,9):
    #        x,y,w,h=cv2.boundingRect(e[i])
    #        u = int(x+0.2*w)
    #        v = int(y+0.2*h)
    #        a = int(x+0.8*w)
    #        b = int(y+0.8*h)
    #        cv2.rectangle(frame,(u,v),(a,b),(0,0,0),2)
    #else:
    #    pass

    #回傳滑鼠位置
    #def position(event,x,y,flags,params):
    #if event==cv2.EVENT_LBUTTONDBLCLK:
    #    print(x,',',y)
    #cv2.setMouseCallback('The colors',position)

    # if 測到9個 (獲得玩整的一面) 將輪廓append到陣列裡
    #先判定是否有重複 沒有then暫停畫面0.4sec 顯示在另一個視窗上
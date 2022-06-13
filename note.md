# OpenCV

## 高斯模糊

>[參考網站 1](https://blog.csdn.net/weixin_39851887/article/details/110899775)  
[參考網站 2](http://www.ruanyifeng.com/blog/2012/11/gaussian_blur.html)

1. 去除雜訊  
2. 利用圖像算法對像素色值進行重新合成的一種處理
3. 每個像素值由其周圍的像素質加權平均所得
4. 均值模糊
5. 線性代數 捲積演算法
6. 語法
+ cv2.GaussianBlur(<font color=#FF0000> intial frame </font>,<font color=#00FFFF> (x,y) </font>,<font color=#00FF00> 0 </font>)  
+ <font color=#00FFFF> (x,y) </font>-> 高斯核大小, xy可以不同, 必須是奇數  
+ <font color=#00FF00>0</font>-> 高斯核的xy方向標準差，0為根據高斯核大小來計算

## 顏色空間轉換

1. 語法
+ cv2.Color(<font color=#FF0000> intial frame </font>,<font color=#00FFFF> 顏色轉換 </font>)
+ <font color=#00FFFF> 顏色轉換 </font>
    >BGR->gray : cv2.COLOR_BGR2GRAY  
    >BGR->HSV  : cv2.COLOR_BGR2HSV

## 邊緣偵測

1. Canny 只能處理灰階影像
2. 語法

## 滑動條

```py
def no(x):
    pass

cv2.namedWindow('視窗名稱')
cv2.createTrackbar('滑動條名稱','視窗名稱',預設值,最大值,no)
a = cv2.getTrackbarPos('滑動條名稱','視窗名稱') #變數a

```

### 
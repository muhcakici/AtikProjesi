import cv2 as cv
import numpy as np

#Gri arka ekranda iyi performans verir
cap=cv.VideoCapture(0)
#wabcam yakalama

while(True):
    ret,frame=cap.read()
    #wabcam okuma

#PLASTİK

    hsv=cv.cvtColor(frame, cv.COLOR_BGR2HSV)
#hsv donusum
    low_blue=np.array([94,160,2])
    upper_blue=np.array([126,255,255])
#maske sinirlari

    blue_mask=cv.inRange(hsv, low_blue, upper_blue)
#maskeleme
    median=cv.medianBlur(blue_mask, 5)
#gürültü giderme
    cnts,hierarchy=cv.findContours(median, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    for cnt in cnts:
        x,y,w,h=cv.boundingRect(cnt)
        area=cv.contourArea(cnt)
        if(150<area<2000):
            cv.putText(frame, "Plastik", (x+20,y+20), cv.FONT_HERSHEY_SIMPLEX, 0.75, (23,197,254),2)


#KAGIT

    gray=cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
#griye donusum
    median2=cv.medianBlur(gray, 5)
#gurultu giderme
    kernel=np.ones((5,5),np.uint8)
#filter icin kernal
    closing=cv.morphologyEx(median2, cv.MORPH_CLOSE, kernel)
#close filter
    cnts,hierarchy=cv.findContours(closing, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    for cnt in cnts:
        x,y,w,h=cv.boundingRect(cnt)
        area=cv.contourArea(cnt)
        if(area>2500):
            cv.putText(frame, "Kagit", (x+30,y+30), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0),2)


#CAM

    median3=cv.medianBlur(frame, 5)
#gurultu giderme
    hsv=cv.cvtColor(median3, cv.COLOR_BGR2HSV)
#hsv donusum
    low_green=np.array([40,80,80])
    upper_green=np.array([80,255,255])
    green_mask=cv.inRange(hsv, low_green, upper_green)
    median4=cv.medianBlur(green_mask, 5)
    kernel=np.ones((5,5),np.uint8)
    closing=cv.morphologyEx(median4, cv.MORPH_CLOSE, kernel)
    cnts,hierarchy=cv.findContours(closing, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    for cnt in cnts:
        x,y,w,h=cv.boundingRect(cnt)
        area=cv.contourArea(cnt)
        if(area>1500):
            cv.putText(frame, "Cam", (x,y+60), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,255),2)

    cv.imshow("Frame", frame)

    k=cv.waitKey(1)
    if k%256 == 27: #esc
        break

cap.release()
cv.destroyAllWindows()

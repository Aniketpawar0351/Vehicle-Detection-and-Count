import cv2

import numpy as np
import matplotlib.pyplot as plt

#webcam
cap = cv2.VideoCapture('video.mp4') #to read video frome directory


min_width_react =80  #min width reactanguler
min_hieght_react =80

count_line_position = 550
#initialize substractor

algo = cv2.bgsegm.createBackgroundSubtractorMOG()


def center_handle(x,y,w,h):
    x1=int(w/2)
    y1=int(h/2)
    cx=x+x1
    cy=y+y1
    return cx,cy

detect = []
detect2 = []
offset=6 #allowble error between pixel
counter =0
counter2 =0



while True: #used to continuos play our frame
    ret,frame1= cap.read()
    grey = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY) # converting to grey
    blur = cv2.GaussianBlur(grey,(3,3),5)
    #applying on each frame
    img_sub = algo.apply(blur)
    dilat = cv2.dilate(img_sub,np.ones((5,5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    dilatada = cv2.morphologyEx(dilat,cv2.MORPH_CLOSE,kernel)
    dilatada = cv2.morphologyEx(dilatada, cv2.MORPH_CLOSE, kernel)
    counterShape,h = cv2.findContours(dilatada,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cv2.imshow('Detector', dilatada)  # this fun show the binirise video

    cv2.line(frame1,(25,count_line_position),(1200,count_line_position),(255,127,0),3)

    for (i,c) in enumerate(counterShape):
        (x,y,w,h) = cv2.boundingRect(c)
        validate_counter = (w>= min_width_react) and (h>= min_hieght_react)
        if not validate_counter:
            continue
        if x>10 and x<600:
            cv2.rectangle(frame1,(x,y),(x+w,y+h),(255,255,0),2)
            cv2.putText(frame1, "VEHICLE " + str(counter), (x,y-20), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 0, 255), 2)

            center =center_handle(x,y,w,h)
            detect.append(center) #list
            cv2.circle(frame1,center,4,(0,0,255),-1)

            for(x,y) in detect:
                if y<(count_line_position+offset) and y>(count_line_position-offset):
                    counter+=1
                    cv2.line(frame1,(25,count_line_position),(1200,count_line_position),(0,127,255),3)
                    detect.remove((x,y))
                    print("vehicle Counter:"+str(counter))

        else:
            cv2.rectangle(frame1,(x,y),(x+w,y+h),(255,255,0),2)
            cv2.putText(frame1, "VEHICLE " + str(counter2), (x,y-20), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 0, 255), 2)

            center2 =center_handle(x,y,w,h)
            detect2.append(center2) #list
            cv2.circle(frame1,center2,4,(0,0,255),-1)

            for(x,y) in detect2:
                if y<(count_line_position+offset) and y>(count_line_position-offset):
                    counter2+=1
                    cv2.line(frame1,(25,count_line_position),(1200,count_line_position),(0,127,255),3)
                    detect2.remove((x,y))
                    print("vehicle Counter:"+str(counter2))

    cv2.putText(frame1,"VEHICLE COUNTER 1:"+str(counter),(50,70),cv2.FONT_HERSHEY_SIMPLEX,1.2,(0,0,255),3)
    cv2.putText(frame1, "VEHICLE COUNTER 2:" + str(counter2), (800, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)


    cv2.imshow('Video frame',frame1) #this fun show the video



    if cv2.waitKey(1)== 13:
        break




cv2.destroyWindow()
cap.release()

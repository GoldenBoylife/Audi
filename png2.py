#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2
import numpy as np
import matplotlib.pylab as plt
##파일 경로로 실행, 동영상 파일 640x320 , 햰뜰png
video_file = "/home/goldenboy/Pictures/kmu_track(mp4)_fast.mp4"
png_file = "/home/goldenboy/Pictures/steer_arrow.png"
#파일 경로로 실행, 동영상 파일 640x320 
#print(video_file.shape)
cap= cv2.VideoCapture(video_file)   #캡처 객체 생성,Gray
#비디오 프레임 하나하나 캡첫해서 cap으로 저장 
png=cv2.imread(png_file, cv2.IMREAD_ANYCOLOR)
######################
def draw_steer(img, steer_angle):
    global Width, Height, arrow_pic
    arrow_pic = cv2.imread('/home/goldenboy/Pictures/steer_arrow.png', cv2.IMREAD_COLOR)
    origin_Height = arrow_pic.shape[0]
    origin_Width = arrow_pic.shape[1]
    steer_wheel_center = origin_Height * 0.74
    arrow_Height = 222/2
    arrow_Width = (arrow_Height * 462)/728
    matrix = cv2.getRotationMatrix2D((origin_Width/2, steer_wheel_center), (steer_angle) * 1.5, 0.7)    
    arrow_pic = cv2.warpAffine(arrow_pic, matrix, (origin_Width+60, origin_Height))
    arrow_pic = cv2.resize(arrow_pic, dsize=(arrow_Width, arrow_Height), interpolation=cv2.INTER_AREA)
    gray_arrow = cv2.cvtColor(arrow_pic, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_arrow, 1, 255, cv2.THRESH_BINARY_INV)
    arrow_roi = img[arrow_Height: 222, (666/2 - arrow_Width/2) : (666/2 + arrow_Width/2)]
    arrow_roi = cv2.add(arrow_pic, arrow_roi, mask=mask)
    res = cv2.add(arrow_roi, arrow_pic)
    img[(222 - arrow_Height): 222, (666/2 - arrow_Width/2): (666/2 + arrow_Width/2)] = res
    cv2.imshow('steer', img)

steer_angle =0
##########################
if cap.isOpened() :                 #캡처 객체 초기화 확인,정상시 True
    while True:
        ret, img = cap.read()
        
        
        #목표박스 만들기
        roi_1= 370
        roi_2=390
        roi =img[roi_1:roi_2 :]
        #gray + blur 효과
        gray= cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)  
        blur = cv2.GaussianBlur(gray, (5,5),0)
        
        ##canny
        low_threshold =50
        high_threshold =200
        edges = cv2.Canny(blur,low_threshold,high_threshold)
        ##팽창  방법1
        #k= cv2.getStructuringElement(cv2.MORPH_RECT,(100,100))
        #구조화 요소 커널, 사각형(3 x 3)생성
        #dila = cv2.dilate(edges,k) 
        
        ##팽창 방법2
        kernel = np.ones((3,3),np.uint8)
        dila = cv2.dilate(edges, kernel, iterations=3)
        
        ##center Red Box
        cv2.rectangle(img,(315,roi_1),(325,roi_2), (0,0,255),2)
        ##Boxcolor
        view =cv2.cvtColor(dila, cv2.COLOR_GRAY2BGR)
        ##nonzero init
        left, right =-1, -1
        ##  left ->right
        for l in range(0,111):  
            #area = dila[5: 15 , l-20 : l]
            area = dila[5: 15 , l-20 : l]
            if cv2.countNonZero(area) > 90:
                left =l
                break
        ##  right -> left
        for r in range(640,518,-1):  #숫자 감소위해 -1
            area =dila[5 :15 , r:r+20]
            if cv2.countNonZero(area) > 90:
                right =r
                break
        ## left 있을 때  
        if left != -1 :
            #lsquare = cv2.rectangle(view,( left- 20 , 5),(left,15),(0,255,0),3)
            lsquare = cv2.rectangle(img,( left- 20 , 5 +roi_1),(left,35+roi_1),(0,255,0),3)
            #img에다가 녹색상자(차선검출) 표시함. roi_1은 y_보정
        
        ## right 있을 때 
        if right != -1:
            #rsquare = cv2.rectangle(view,(right,5),(right+20,15), (0,255,0),3)
            rsquare = cv2.rectangle(img,(right,5+roi_1),(right+20,35+roi_1), (0,255,0),3)
            #roi_1은 y값 보정,  0은 감지상자 높이

        ## if문 left 와 right 둘 다 있을 때-> 그 사이값과 센터와의 거리
        if left !=-1 and right !=-1:
            cent_square= cv2.rectangle(img,((right-20+left)/2+20,5+roi_1),((right+left)/2,15+roi_1),(255,0,0),3)
            steer_angle = (((right+left)/2)-320)/4
            print (steer_angle)
            print("Lost both")
        ## elif문 left 없고, right 있을 때 
        elif left ==-1 and right !=-1:
            cent_square= cv2.rectangle(img,(right-20-300,5+roi_1),(right-300,15+roi_1),(255,0,0),3)
            steer_angle = (360-(right-20-300))/4.0
            #센터와으 거리에 비례해서 
            print (steer_angle)
            if steer_angle >=50:
                steer_angle = 50
            print("Lost left line")

        ## elif문 left 있고, right 없을 때
        elif left !=-1 and right ==-1:
            cent_square= cv2.rectangle(img,(left-20+300,5+roi_1),(left+300,15+roi_1),(255,0,0),3) 
            #차선 가운데인데, 오른쪽 차선에서 약 200정도 떨어져 있음
            steer_angle = (left-20-300)/4.0
            print (steer_angle)
            if steer_angle <=-61:
                steer_angle = 0
            elif steer_angle <= -45:
                steear_angle = -35
            print("Lost right line")

        ## else (둘다 없을 때)
        else :
            steer_angle = steer_angle * 1
 
			
        if ret:
            
            #cv2.imshow("origin",img) #원본 이미지
            cv2.imshow("view", view) #roi 이미지
            #cv2.imshow( video_file,dila)
            
            cv2.waitKey(25)
        else:
            break
        
        draw_steer(img,steer_angle)
else:
    print("can't open video.")
    cap.release()
    cv2.destroyAllWindows()

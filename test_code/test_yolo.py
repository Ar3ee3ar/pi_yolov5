# from importlib.resource import path
from time import time
import torch
from matplotlib import pyplot as plt

import numpy as np
import cv2

import imutils

import logging

import RPi.GPIO as  GPIO


GPIO.setmode(GPIO.BCM) 
led_in_1 = 16
led_in_2 = 20
led_in_3 = 21
GPIO.setup(led_in_1, GPIO.OUT) # LED_1
GPIO.setup(led_in_2, GPIO.OUT) #LED_2
GPIO.setup(led_in_3, GPIO.OUT) #LED_3

shape_stack = []

def send_signal(shape):
	if(shape=="circle"):
		print("here----------2")
		GPIO.output(led_in_1, GPIO.LOW)
		GPIO.output(led_in_2, GPIO.LOW)
		GPIO.output(led_in_3, GPIO.LOW)
	elif(shape=="rectangle"):
		print("here----------3")
		GPIO.output(led_in_1, GPIO.HIGH)
		GPIO.output(led_in_2, GPIO.LOW)
		GPIO.output(led_in_3, GPIO.LOW)
	elif(shape=="square"):
		print("here----------4")
		GPIO.output(led_in_1, GPIO.LOW)
		GPIO.output(led_in_2, GPIO.HIGH)
		GPIO.output(led_in_3, GPIO.LOW)
	elif(shape=="triangle"):
		print("here----------5")
		GPIO.output(led_in_1, GPIO.LOW)
		GPIO.output(led_in_2, GPIO.LOW)
		GPIO.output(led_in_3, GPIO.HIGH)


model = torch.hub.load('ultralytics/yolov5','custom',path='best.pt',force_reload=True)

cap = cv2.VideoCapture('/dev/video0',cv2.CAP_V4L)

cap.set(cv2.CAP_PROP_FRAME_WIDTH,240)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,240)

while cap.isOpened():
    start = time()
    ret, frame = cap.read()
    resized = imutils.resize(frame, width=300)
    mark_resized = resized.copy()
    # change bg color
    hsv=cv2.cvtColor(resized,cv2.COLOR_BGR2HSV)

    # # Define lower and uppper limits of what we call "brown"
    hsv_low = np.array([0, 0, 106], np.uint8)
    hsv_high = np.array([179, 255, 255], np.uint8)

    # # Mask image to only select browns
    mask=cv2.inRange(hsv,hsv_low,hsv_high)

    # # Change image to red where we found brown
    #mark_resized[mask>0]=(0,0,0)
    res_mark = cv2.bitwise_and(mark_resized, mark_resized, mask=mask)
    #
    gray = cv2.cvtColor(res_mark, cv2.COLOR_BGR2GRAY)
    
    # draw reference line
    start_point = ((int(resized.shape[1]/2)), 0) 
    end_point = ( (int(resized.shape[1]/2)), (int(resized.shape[0])) )
    # print('start_point: ',start_point)
    # print('end_point: ',end_point)
    color = (0, 0, 255)
    thickness = 2
    #cv2.line(resized, start_point, end_point, color, thickness)
    cv2.line(resized, ((int(resized.shape[1]/2)-10), 0) , ( (int(resized.shape[1]/2)-10), (int(resized.shape[0])) ), (0, 255, 0), thickness) # green
    cv2.line(resized, ((int(resized.shape[1]/2)-20), 0) , ( (int(resized.shape[1]/2)-20), (int(resized.shape[0])) ), (255, 0, 0), thickness) #blue
    results = model(gray)
    try:
        # print(results.pandas().xyxy[0])
        result_array = (results.pandas().xyxy[0]).to_numpy()
        for i in range(len(result_array)):
            if(result_array[i][4] > 0.2):
                top_left = (int(result_array[i][0]),int(result_array[i][1]))
                bottom_right = (int(result_array[i][2]),int(result_array[i][3]))
                # for counting
                cx = int(round((result_array[i][0]+result_array[i][2])//2))
                cy = int(round((result_array[i][1]+result_array[i][3])//2))
                shape = str(result_array[i][6])
                if(cx <= (int((resized.shape[1]/2))+10) and cx >= (int((resized.shape[1]/2))-10)):
                    if(shape== "circle"):
                        print('circle')
                        send_signal('circle')
                    elif(shape == "triangle"):
                        print('triangle')
                        send_signal('triangle')
                    elif(shape == "square"):
                        print('square')
                        send_signal('square')
                    elif(shape == "rectangle"):
                        print('rectangle')
                        send_signal('rectangle')
                    print(result_array[i][4])
                org_text = (int(result_array[i][0]),int(result_array[i][1])-10)
                cv2.circle(resized,(cx,cy),2,(255,0,0),2)
                cv2.rectangle(resized,top_left,bottom_right,(0,0,255),2)
                cv2.putText(resized,str(result_array[i][6]),org_text,cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2,cv2.LINE_AA)
    except Exception as e: # work on python 3.x
    	print(str(e))
    cv2.imshow('Screen', resized)
    cv2.imshow('mark Screen', res_mark)
    if(cv2.waitKey(1) & 0xFF == ord("q")) or (cv2.waitKey(1)==27):
        break
    if(cv2.getWindowProperty("Screen", cv2.WND_PROP_VISIBLE) < 1):
        break

    end = time()
    sec = end-start
    #print(sec)

cap.release()
cv2.destroyAllWindows()

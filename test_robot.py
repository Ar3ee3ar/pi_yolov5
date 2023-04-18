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

obj_sensor = 27
initialize = 17

GPIO.setup(led_in_1, GPIO.OUT) # LED_1
GPIO.setup(led_in_2, GPIO.OUT) #LED_2
GPIO.setup(led_in_3, GPIO.OUT) #LED_3

GPIO.setup(obj_sensor,GPIO.IN) # obj_sensor
GPIO.setup(initialize,GPIO.IN) # obj_sensor

shape_stack = []
prev_state = 0
time_start_send_signal = 0
already_send = False
prev_cy = 0
prev_frame = 0
def send_signal(shape):
	if(shape=="initial"):
		GPIO.output(led_in_1, GPIO.HIGH)
		GPIO.output(led_in_2, GPIO.LOW)
		GPIO.output(led_in_3, GPIO.HIGH)
	elif(shape=="circle"):
		GPIO.output(led_in_1, GPIO.LOW)
		GPIO.output(led_in_2, GPIO.LOW)
		GPIO.output(led_in_3, GPIO.LOW)
	elif(shape=="rectangle"):
		GPIO.output(led_in_1, GPIO.HIGH)
		GPIO.output(led_in_2, GPIO.LOW)
		GPIO.output(led_in_3, GPIO.LOW)
	elif(shape=="square"):
		GPIO.output(led_in_1, GPIO.LOW)
		GPIO.output(led_in_2, GPIO.HIGH)
		GPIO.output(led_in_3, GPIO.LOW)
	elif(shape=="triangle"):
		GPIO.output(led_in_1, GPIO.LOW)
		GPIO.output(led_in_2, GPIO.LOW)
		GPIO.output(led_in_3, GPIO.HIGH)
	elif(shape=="wait"):
		GPIO.output(led_in_1, GPIO.HIGH)
		GPIO.output(led_in_2, GPIO.HIGH)
		GPIO.output(led_in_3, GPIO.HIGH)
		
		
model = torch.hub.load('ultralytics/yolov5','custom',path='best.pt',force_reload=False)

cap = cv2.VideoCapture('/dev/video0',cv2.CAP_V4L)

#cap.set(cv2.CAP_PROP_FRAME_WIDTH,1000)
cv2.namedWindow('Screen',cv2.WINDOW_NORMAL)
#cv2.setWindowProperty('Screen',cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)

send_signal('wait')
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

    # # Change belt conveyer's color to black
    res_mark = cv2.bitwise_and(mark_resized, mark_resized, mask=mask)
    #
    gray = cv2.cvtColor(res_mark, cv2.COLOR_BGR2GRAY)
    
    # draw reference line
    start_point = (0,(int(resized.shape[0]/2))) 
    end_point = ( (int(resized.shape[1])), (int(resized.shape[0]/2)) )
    color = (0, 0, 255)
    thickness = 2
    #cv2.line(resized, start_point, end_point, color, thickness)
    cv2.line(resized, (70,(int(resized.shape[0]/2)+10)) , ((int(resized.shape[1])-80),(int(resized.shape[0]/2)+10) ), (0, 255, 0), thickness) # green
    cv2.line(resized, (70,(int(resized.shape[0]/2)-10)) , ((int(resized.shape[1])-80),(int(resized.shape[0]/2)-10)), (255, 0, 0), thickness) #blue
    #print(GPIO.input(obj_sensor))
    #print('---------------initial : ',GPIO.input(initialize))
    if(GPIO.input(initialize) == 0):
    	#print('initial')
    	send_signal('wait')
    	time_start_send_signal = time()
    	already_send = True
    if(GPIO.input(obj_sensor) == 0):
    	if(time() - prev_frame >1.0):
    		prev_frame = time()
    		prev_state = GPIO.input(obj_sensor)
    		results = model(gray)
    		print(results)
    		try:
    			#print (len((results.pandas().xyxy[0]).to_numpy()))
    			result_array = (results.pandas().xyxy[0]).to_numpy()
    			if(len(result_array) > 0):
    				for i in range(len(result_array)):
    					if(result_array[i][4] > 0.6):
    						top_left = (int(result_array[i][0]),int(result_array[i][1]))
    						bottom_right = (int(result_array[i][2]),int(result_array[i][3]))
    						# for counting
    						cx = int(round((result_array[i][0]+result_array[i][2])//2))
    						cy = int(round((result_array[i][1]+result_array[i][3])//2))
    						if((cy <= (int((resized.shape[0]/2))+10) and cy >= (int((resized.shape[0]/2))-10)) and (cx>=65 and cx<=(int(resized.shape[1])-75))):
    							if(((cy!= prev_cy and (cy!=prev_cy+1 and cy!=prev_cy-1)) or prev_cy==0)):
	    							shape = str(result_array[i][6])
	    							#print(shape,': prev_cy = ',prev_cy,' | cy= ',cy)
	    							#print(result_array[i][4])
	    							shape_stack.append(shape)
	    							#print('prev_cy = ',prev_cy,' | cy= ',cy)
	    							print('stored: ',shape,' => ',shape_stack)
	    							prev_cy = cy
	    						org_text = (int(result_array[i][0]),int(result_array[i][1])-10)
	    						cv2.circle(resized,(cx,cy),2,(255,0,0),2)
	    						cv2.rectangle(resized,top_left,bottom_right,(0,0,255),2)
	    						cv2.putText(resized,str(result_array[i][6])+str(cy),org_text,cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2,cv2.LINE_AA)
    				cv2.imshow('Screen', resized)
    			else:
    				print('wait')
    				send_signal('wait')
    		except Exception as e: # work on python 3.x
    			print(str(e))
    	elif(time() - prev_frame >1.5):
    		print('stop')
    		pass
    elif(GPIO.input(obj_sensor) == 1 and prev_state == 0):
    	prev_state = GPIO.input(obj_sensor)
    	if(len(shape_stack)>0):
    		want_shape = shape_stack.pop(0)
    		print('want_shape : ',want_shape,'')
    		send_signal(want_shape)
    		time_start_send_signal = time()
    		already_send = True
    	else:
    		print('no shape')
    		send_signal('wait')
    #print('obj: ',GPIO.input(obj_sensor),'--------------------')
    #cv2.imshow('mark Screen', res_mark)
    if(already_send and ((time() - time_start_send_signal) > 5)):
    	#print('stop send shape signal')
    	send_signal('wait')
    	already_send = False
    #elif(already_send and ((time() - time_start_send_signal) <= 5)):
    	#print('still sent')
    #if((time() - time_start_send_signal) <= 5):
    	#print('shape signal')
    if(cv2.waitKey(1) & 0xFF == ord("q")) or (cv2.waitKey(1)==27):
        break
    if(cv2.getWindowProperty("Screen", cv2.WND_PROP_VISIBLE) < 1):
        break

    end = time()
    sec = end-start
    #print(sec)

cap.release()
cv2.destroyAllWindows()

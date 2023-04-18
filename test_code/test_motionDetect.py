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

def diffImg(t0, t1, t2):
	d1 = cv2.absdiff(t2, t1)
	d2 = cv2.absdiff(t1, t0)
	return cv2.bitwise_and(d1, d2)

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

# Read three images first:
t_minus = cv2.cvtColor(cap.read()[1], cv2.COLOR_RGB2GRAY)
t = cv2.cvtColor(cap.read()[1], cv2.COLOR_RGB2GRAY)
t_plus = cv2.cvtColor(cap.read()[1], cv2.COLOR_RGB2GRAY)

detect_count = 0

while cap.isOpened():
	cv2.imshow( "Movement Indicator", diffImg(t_minus, t, t_plus) )
	# Read next image
	if(np.amax(diffImg(t_minus, t, t_plus)) > 100):
		print('detect movement : '+str(detect_count))
		detect_count = detect_count+1
	t_minus = t
	t = t_plus
	t_plus = cv2.cvtColor(cap.read()[1], cv2.COLOR_RGB2GRAY)
	key = cv2.waitKey(10)
	if key == 27:
		cv2.destroyWindow("Movement Indicator")
		break

print("Goodbye")


    

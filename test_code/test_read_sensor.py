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

def send_signal(shape):
	if(shape=="initial"):
		GPIO.output(led_in_1, GPIO.HIGH)
		GPIO.output(led_in_2, GPIO.HIGH)
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
	elif(shape=="pull"):
		GPIO.output(led_in_1, GPIO.LOW)
	elif(shape=="push"):
		GPIO.output(led_in_1, GPIO.HIGH)
		
while(True):
	print(GPIO.input(initialize))
		
		


# from importlib.resource import path
from time import time
import torch
from matplotlib import pyplot as plt

import numpy as np
import cv2

import imutils

import logging

import argparse
import json

from models.common import DetectMultiBackend
from utils.dataloaders import LoadStreams
from utils.general import (check_img_size, check_imshow, non_max_suppression, scale_boxes,Profile,cv2)
from utils.torch_utils import select_device
from utils.plots import Annotator,colors

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

#shape_class = ['circle','rectangle','square','triangle']

def _argparse():
    # print('parsing args...')
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", "-file_name", type=str, default='class.json', help="file name") 
    parser.add_argument("--weight", "-file_weight",type=str, default='best.onnx', help="file weight name")
    parser.add_argument("--imgsz", "-image_size", type=int, default='240', help="image size")
    arg = parser.parse_args()
    return arg

def send_signal(shape):
	if(shape=="initial"):
		GPIO.output(led_in_1, GPIO.HIGH)
		GPIO.output(led_in_2, GPIO.LOW)
		GPIO.output(led_in_3, GPIO.HIGH)
	elif(shape=="wait"):
		GPIO.output(led_in_1, GPIO.HIGH)
		GPIO.output(led_in_2, GPIO.HIGH)
		GPIO.output(led_in_3, GPIO.HIGH)
	else:
		GPIO.output(led_in_1, class_signal[shape][0])
		GPIO.output(led_in_2, class_signal[shape][0])
		GPIO.output(led_in_3, class_signal[shape][0])
		
file_name = _argparse().file
file_weight = _argparse().weight
image_size = _argparse().imgsz

with open(file_name) as json_file:
    class_signal = json.load(json_file)
    shape_class = list(class_signal.keys())
		
#model = torch.hub.load('ultralytics/yolov5','custom',path='best.pt',force_reload=False)

#cap = cv2.VideoCapture('/dev/video0',cv2.CAP_V4L)

#cap.set(cv2.CAP_PROP_FRAME_WIDTH,1000)
cv2.namedWindow('Screen',cv2.WINDOW_NORMAL)
#cv2.setWindowProperty('Screen',cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)

send_signal('wait')

#ret, frame = cap.read()
#resized = imutils.resize(frame, width=300)
#mark_resized = resized.copy()
## change bg color
#hsv=cv2.cvtColor(resized,cv2.COLOR_BGR2HSV)

# # Define lower and uppper limits of what we call "brown"
#hsv_low = np.array([0, 0, 106], np.uint8)
#hsv_high = np.array([179, 255, 255], np.uint8)

# # Mask image to only select browns
#mask=cv2.inRange(hsv,hsv_low,hsv_high)

# # Change belt conveyer's color to black
#res_mark = cv2.bitwise_and(mark_resized, mark_resized, mask=mask)
#
#gray = cv2.cvtColor(res_mark, cv2.COLOR_BGR2GRAY)

# draw reference line
#start_point = (0,(int(resized.shape[0]/2))) 
#end_point = ( (int(resized.shape[1])), (int(resized.shape[0]/2)) )
#color = (0, 0, 255)
#thickness = 2
#cv2.line(resized, start_point, end_point, color, thickness)
#cv2.line(resized, (70,(int(resized.shape[0]/2)+10)) , ((int(resized.shape[1])-80),(int(resized.shape[0]/2)+10) ), (0, 255, 0), thickness) # green
#cv2.line(resized, (70,(int(resized.shape[0]/2)-10)) , ((int(resized.shape[1])-80),(int(resized.shape[0]/2)-10)), (255, 0, 0), thickness) #blue
#print(GPIO.input(obj_sensor))
#print('---------------initial : ',GPIO.input(initialize))

#------------------- add from detect.py ------------------------------
device = select_device('cpu')
model = DetectMultiBackend(weights=file_weight, device=device)
stride, names, pt = model.stride, model.names, model.pt
imgsz = check_img_size(imgsz=(image_size,image_size), s=stride)  # check image size

# Dataloader
bs = 1  # batch_size
view_img = check_imshow(warn=True)
dataset = LoadStreams(sources='0', img_size=imgsz, stride=stride, auto=pt, vid_stride=1) # change source to pi camera
bs = len(dataset)
vid_path, vid_writer = [None] * bs, [None] * bs

# Run inference
model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
for path, im, im0s, vid_cap, s in dataset:
    # if(True):
    with dt[0]:
    	im = torch.from_numpy(im).to(model.device)
    	im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
    	im /= 255  # 0 - 255 to 0.0 - 1.0
    	if len(im.shape) == 3:
    		im = im[None]  # expand for batch dim
    # Inference
    with dt[1]:
    # visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
    	pred = model(im, augment=False, visualize=False)
    
    # NMS
    with dt[2]:
    	pred = non_max_suppression(pred, conf_thres=0.45, iou_thres=0.45, classes=None, max_det=1000)
    	
    for i, det in enumerate(pred):  # per image
    	p, im0, frame = path[i], im0s[i].copy(), dataset.count
    	annotator = Annotator(im0, line_width=3, example=str(names))
    	if len(det):
    		det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
    		# Print results
    		for c in det[:, 5].unique():
    			n = (det[:, 5] == c).sum()  # detections per class
    			s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
    		for *xyxy, conf, cls in reversed(det):
    			c = int(cls)  # integer class
    			label = f'{names[c]} {conf:.2f}'
    			annotator.box_label(xyxy, label, color=colors(c, True))
    	# #---------------------------------------------------------------------
    	#results = model(gray)
    	#print(results)
    	result_array = list(det.numpy())
    	# draw reference line
    	start_point = (0,(int(im0.shape[0]/2))) 
    	end_point = ( (int(im0.shape[1])), (int(im0.shape[0]/2)) )
    	color = (0, 0, 255)
    	thickness = 2
    	#cv2.line(resized, start_point, end_point, color, thickness)
    	cv2.line(im0, (70,(int(im0.shape[0]/2)+10)) , ((int(im0.shape[1])-80),(int(im0.shape[0]/2)+10) ), (0, 255, 0), thickness) # green
    	cv2.line(im0, (70,(int(im0.shape[0]/2)-10)) , ((int(im0.shape[1])-80),(int(im0.shape[0]/2)-10)), (255, 0, 0), thickness) #blue
    	try:
    		#print(result_array)
    		#print (len((results.pandas().xyxy[0]).to_numpy()))
    		#result_array = (results.pandas().xyxy[0]).to_numpy()
    		if(len(result_array) > 0):
    			for i in range(len(result_array)):
    				if(result_array[i][4] > 0.4):
    					top_left = (int(result_array[i][0]),int(result_array[i][1]))
    					bottom_right = (int(result_array[i][2]),int(result_array[i][3]))
    					# for counting
    					cx = int(round((result_array[i][0]+result_array[i][2])//2))
    					cy = int(round((result_array[i][1]+result_array[i][3])//2))
    					shape = shape_class[int(result_array[i][5])]
    					if((cy <= (int((im0.shape[0]/2))+10) and cy >= (int((im0.shape[0]/2))-10)) and (cx>=65 and cx<=(int(im0.shape[1])-75))):
    						if(((cy!= prev_cy and (cy!=prev_cy+1 and cy!=prev_cy-1)) or prev_cy==0)):
    							#shape = str(result_array[i][6])
    							#print(shape,': prev_cy = ',prev_cy,' | cy= ',cy)
    							#print(result_array[i][4])
    							shape_stack.append(shape)
    							#print('prev_cy = ',prev_cy,' | cy= ',cy)
    							print('stored: ',shape,' => ',shape_stack)
    							prev_cy = cy
    						org_text = (int(result_array[i][0]),int(result_array[i][1])-10)
    					cv2.circle(im0,(cx,cy),2,(255,0,0),2)
    						#cv2.rectangle(resized,top_left,bottom_right,(0,0,255),2)
    						#cv2.putText(resized,str(result_array[i][6])+str(cy),org_text,cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2,cv2.LINE_AA)
    		else:
    			print('wait')
    			send_signal('wait')
    	except Exception as e: # work on python 3.x
    		print(str(e))
    	im0 = annotator.result()
    	cv2.imshow('Screen', im0)
    	# check if intialize case ---------------
    	if(GPIO.input(initialize) == 0):
    		#print('initial')
    		send_signal('wait') # change to initial
    		time_start_send_signal = time()
    		already_send = True
    	#----------------------------------------------
    	if(GPIO.input(obj_sensor) == 0):
    		#print('obj: ',GPIO.input(obj_sensor),'--------------------')
    		prev_frame = time()
    		prev_state = GPIO.input(obj_sensor)
    	# print(f"{dt[1].dt * 1E3:.1f}ms") # print time
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
    	# control when pi need to stop send signal to robot -----------------------
    	if(already_send and ((time() - time_start_send_signal) > 5)):
    		#print('stop send shape signal')
    		send_signal('wait')
    		already_send = False
    		#elif(already_send and ((time() - time_start_send_signal) <= 5)):
    		#print('still sent')
    		#if((time() - time_start_send_signal) <= 5):
    		#print('shape signal')
    	# -------------------------------------------------------------------
    	#end = time()
    	#sec = end-start
    	#print(sec)
cv2.waitKey(1)  # 1 millisecond

#cap.release()
#cv2.destroyAllWindows()

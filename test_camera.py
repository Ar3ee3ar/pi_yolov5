
import cv2

cap = cv2.VideoCapture(-1,2)

ret,frame = cap.read()
print(ret)
cv2.imwrite('image.jpg',frame)

cap.release()

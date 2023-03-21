
import cv2

cap = cv2.VideoCapture('/dev/video0',cv2.CAP_V4L)
i = 0
while cap.isOpened():
    ret,frame = cap.read()
	##cv2.imwrite('image.jpg',frame)
    cv2.putText(frame,'test',(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2,cv2.LINE_AA)
    cv2.imshow('Screen', frame)
    if(cv2.waitKey(1) & 0xFF == ord("a")):
        print('click')
        cv2.imwrite('test_img/image_'+str(i)+'.jpg',frame)
        i = i+1
    # print(fps)

cap.release()
cv2.destroyAllWindows()

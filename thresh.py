import cv2
import numpy as np
import mahotas

cap = cv2.VideoCapture("myvideo.mp4")
while (cap.isOpened()):
                ret,frame = cap.read()
                if ret is False:
                        break
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		blurred = cv2.GaussianBlur(image,(5,5),0)
		threshold_value = mahotas.thresholding.otsu(blurred)


		thresh = image.copy()
		thresh[thresh >threshold_value] = 255
		thresh[thresh < 255] = 0
		#thresh = cv2.bitwise_not(thresh)
		
		frame_copy = frame.copy()
		_, contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)


		contours = sorted(contours, key = cv2.contourArea, reverse = True)[:2] # get two largest contour

		for i in range(0,2):
			x,y,w,h = cv2.boundingRect(contours[i])
			cv2.rectangle(frame_copy,(x,y),(x+w,y+h),(0,255,0),2)	
	
		#for contour in contours:
            		#cv2.drawContours(frame_copy, contour, -1, (0, 255, 0), 7)

		cv2.imshow("original_image", frame)
		cv2.imshow("mask_image",thresh)
		cv2.imshow("filtered_image", frame_copy)
	
                if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
cap.release()
cv2.destroyAllWindows()


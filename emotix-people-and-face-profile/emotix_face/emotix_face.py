import numpy as np
import cv2 as cv
import os, sys
cnt=0
cascade_path="face_cascades/haarcascade_profileface.xml"
face_cascade = cv.CascadeClassifier(cascade_path)
path = "../Detected Subjects/"
dirs = os.listdir(path)
for item in dirs:
        if os.path.isfile(path+item):
			img = cv.imread(path+item)
			gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
			faces = face_cascade.detectMultiScale(gray,  1.1, 2, 0, (20, 20))
			for (x,y,w,h) in faces:
				crop_img = img[y:y+h,x:x+w]
				cv.imwrite("../Detected Subjects/Faces/subject_face"+str(cnt)+".png", crop_img)
				cnt +=1
				cv.imshow('img',crop_img)
cv.waitKey(0)
cv.destroyAllWindows()


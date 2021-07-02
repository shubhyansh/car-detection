#!C:/Users/91971/anaconda3/python
#!C:/Users/91971/anaconda3/Library\mingw-w64/bin
#!C:/Users/91971/anaconda3/Library\usr/bin
#!C:/Users/91971/anaconda3/Library/bin
print()

import cgi
import subprocess as sb
import os
f = cgi.FieldStorage()
cmd= f.getvalue("x")
import cv2

cap = cv2.VideoCapture(cmd)

car_cascade = cv2.CascadeClassifier('cars.xml')


while True:
    ret, frames = cap.read()
    detect=frames
    gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
    cars = car_cascade.detectMultiScale( frames, 1.3, 5)
    for (x,y,w,h) in cars:
        
        cv2.imshow('Car Detection', frames)
        crop=frames[y-100:y+h+100,x-100:x+w+100]

    if cv2.waitKey(1000) == 13:
        cv2.imwrite("car1.jpg",crop)
        cv2.rectangle(detect,(x,y),(x+w,y+h),(0,0,255),2)
        cv2.rectangle(detect,(x-100,y-250),(x+w+50,y+h+100),(0,255,0),2)
        detect=detect[0:500,0:500]
        cv2.imwrite("D:/talonts/xampp/htdocs/javascript/display.jpg",detect)
        break

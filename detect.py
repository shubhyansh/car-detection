#!C:/Users/91971/anaconda3/python
#!C:/Users/91971/anaconda3/Library\mingw-w64/bin
#!C:/Users/91971/anaconda3/Library\usr/bin
#!C:/Users/91971/anaconda3/Library/bin
import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
import cgi
import os
f = cgi.FieldStorage()
cmd= f.getvalue("x")

data_path = './images/'
onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]

Training_Data, Labels = [], []

for i, files in enumerate(onlyfiles):
    image_path = data_path + onlyfiles[i]
    images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    Training_Data.append(np.asarray(images, dtype=np.uint8))
    Labels.append(i)

Labels = np.asarray(Labels, dtype=np.int32)

vimal_model  = cv2.face_LBPHFaceRecognizer.create()
vimal_model.train(np.asarray(Training_Data), np.asarray(Labels))
print("Model trained sucessefully")
cap = cv2.VideoCapture(cmd)
ret, frames = cap.read()
face = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
results = vimal_model.predict(face)
if results[1] < 500:
    confidence = int( 100 * (1 - (results[1])/400) )
    display_string = str(confidence) + '% Confident it is User'
    if confidence>= 85:
        cv2.imshow('carseen Recognition', face )
        print ('car detected')
        detector(cmd)
    else:
        print("NO car detected")
       
cv2.waitKey() == 13
cv2.destroyAllWindows() 

cv2.destroyAllWindows()
def detector(cmd):
    import cv2

    cap = cv2.VideoCapture(cmd)

    car_cascade = cv2.CascadeClassifier('cars.xml')

    while True:
        ret, frames = cap.read()
        gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
        cars = car_cascade.detectMultiScale(gray, 1.1, 1)
        for (x,y,w,h) in cars:
            cv2.rectangle(gray,(x,y),(x+w,y+h),(0,0,255),2)
            cv2.imshow('Car Detection', gray)
            cv2.imwrite('photo.jpeg', gray)
            break

        if cv2.waitKey(1) == 13:
            break

        
    cv2.destroyAllWindows() 
    import cv2
    import imutils
    import numpy as np
    import pytesseract
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

    img = cv2.imread('./car1.jpg',cv2.IMREAD_COLOR)
    img = cv2.resize(img, (600,400) )

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    gray = cv2.bilateralFilter(gray, 13, 15, 15) 

    edged = cv2.Canny(gray, 30, 200) 
    contours = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)[:10]
    screenCnt = None

    for c in contours:
    
        peri = cv2.arcLength(c, True)
       approx = cv2.approxPolyDP(c, 0.018 * peri, True)
 
        if len(approx) == 4:
          screenCnt = approx
         break

    if screenCnt is None:
        detected = 0
        print ("No contour detected")
    else:
         detected = 1

    if detected == 1:
        cv2.drawContours(img, [screenCnt], -1, (0, 0, 255), 3)

    mask = np.zeros(gray.shape,np.uint8)
    new_image = cv2.drawContours(mask,[screenCnt],0,255,-1,)
    new_image = cv2.bitwise_and(img,img,mask=mask)

    (x, y) = np.where(mask == 255)
    (topx, topy) = (np.min(x), np.min(y))
    (bottomx, bottomy) = (np.max(x), np.max(y))
    Cropped = gray[topx:bottomx+1, topy:bottomy+1]

    text = pytesseract.image_to_string(Cropped, config='--psm 11')
    print("programming_fever's License Plate Recognition\n")
    print("Detected license plate Number is:",text)
    img = cv2.resize(img,(500,300))
    Cropped = cv2.resize(Cropped,(400,200))
    cv2.imshow('car',img)
    cv2.imshow('Cropped',Cropped)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    file = open('read.txt', 'rw')
    file.write(text)
    file.close()
    import requests
    import json
    import xmltodict

    vehicle_reg_no =y #insert the correct registration number
    username = "miss" #insert your user name

    url = "http://www.regcheck.org.uk/api/reg.asmx/CheckIndia?RegistrationNumber=" + vehicle_reg_no +"&username="+username
    url=url.replace(" ","%20")
    print(url)

    r = requests.get(url)
    n = xmltodict.parse(r.content)
    k = json.dumps(n)
    df = json.loads(k)
    l=df["Vehicle"]["vehicleJson"]
    p=json.loads(l)
    x=(str(p['Owner'])+"\n"+str(p['Description'])+"\n"+str(p['RegistrationYear'])+"\n"+str(p['Location'])+"\n"+str(p['FuelType']['CurrentTextValue']))
    file = open('read.txt', 'rw')
    file.write(text)
    file.close()
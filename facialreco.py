import cv2 as cv 
import numpy as np 
import pickle
import insert
import smtplib 
import os
from email.message import EmailMessage
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
import imghdr as igr 



vid = cv.VideoCapture(0)

facecas = cv.CascadeClassifier('haarcascades/data/haarcascade_frontalface_alt2.xml')
recog = cv.face.LBPHFaceRecognizer_create()

recog.read("trainer.yml")


labels={}




with open("pickled.pickle", "rb") as f:
    orglabels = pickle.load(f)
    labels = {v:k for k,v in orglabels.items()}







while True:

    red,frame = vid.read()
    gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    faces =facecas.detectMultiScale(gray,scaleFactor = 1.1,minNeighbors=3)


    for x,y,w,h in faces:  # region of interest
        
        roi = gray[y:y+h,x:x+w]


        id_,confidence = recog.predict(roi) # predicting

        if confidence>45 and confidence<85:
            print(id_,labels[id_])

            font = cv.FONT_HERSHEY_COMPLEX
            name = labels[id_]
            print(name ,"is here and the owner of this device ")
            
            stroke = 2
            color = (20,25,255)

            
                
                
                
                


            cv.putText(frame,name,(x,y),font,1,color,stroke,cv.LINE_AA)
        else:
            print("nothing worng")
            imgi = "theif.png"
            cv.imwrite(imgi, roi)
            
            print("photo clicked")
            facepic = cv.imread("theif.png")
            filepic = imgi
            with open(filepic, "rb") as f:
                photo = f.read()
                imgtype = igr.what(f.name)
                imgname = f.name


            print("photo added")
            insert.add(photo)
            server =smtplib.SMTP_SSL("smtp.gmail.com",465)
            newMessage = EmailMessage()
            
            
            newMessage.add_attachment(photo, maintype='image', subtype=imgtype, filename=imgname)
            server.login("yvverma9@gmail.com","armyismylife")
            server.sendmail("yvverma9@gmail.com","yvverma8@gmail.com",newMessage.as_bytes())
            
            server.quit()

            
        # imgi = "new.png" # saving image of mine
        # cv.imwrite(imgi,roi)

        color = (255,0,0)
        stroke =2

        cv.rectangle(frame,(x,y),(x+w,y+h),color,stroke)
        

    cv.imshow("Yuvraj",frame)

    if( cv.waitKey(20)&0xFF ==ord('q')):
        break

vid.release()
cv.destroyAllWindows()

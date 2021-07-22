import cv2 as cv 
import numpy as np 
import os 
import pickle as pk 


from PIL import Image

basedir = os.path.dirname(os.path.abspath(__file__))
imgdir = os.path.join(basedir,"image")


#face recognizer algo


facecas = cv.CascadeClassifier(
    'haarcascades/data/haarcascade_frontalface_alt2.xml')


recog = cv.face.LBPHFaceRecognizer_create()
xtrain = []
ylabel = []
labelid= {}
currentid=0
for roots,dirs,files in os.walk(imgdir):

    for file in files:

        if(file.endswith("jpg") or file.endswith("png")):

            path = os.path.join(roots,file)
         
            label = os.path.basename(os.path.dirname(path))

            print(label)

            print(path)

           

            # xtrain.append(path)
            # ylabel.append(label)

            # lets convert the images int5o gry6 and convert into numpy array


            if label in labelid:
                pass
            else:
                labelid[label] = currentid
                currentid+=1
            id_ = labelid[label]
             

            pil = Image.open(path).convert("L")

            resize = pil.resize((1000,1000),Image.ANTIALIAS)

            imgarr = np.array(resize,"uint8")

            faces = facecas.detectMultiScale(imgarr,scaleFactor =1.4,minNeighbors=5)

            for x,y,w,h in faces:

                roi = imgarr[y:y+h,x:x+w]
                xtrain.append(roi)
                ylabel.append(id_)


with open("pickled.pickle","wb") as f:
    pk.dump(labelid,f)

recog.train(xtrain,np.array(ylabel))
recog.save("trainer.yml")





print(labelid)


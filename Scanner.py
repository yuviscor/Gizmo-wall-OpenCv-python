import cv2 as cv 
import numpy as np 


# image # gray scale edge contours filer biggest contour warp perspective adaptive threshholding # saving image

import utils1

webcamfee =False
pathimage = "1.jpg"

cap = cv.VideoCapture(0)

cap.set(10,150)
h = 640
w = 480
#################################################

utils1.initializeTrackbars()
count =0

while True:
    Blank = np.zeros((h, w, 3), np.uint8)
    if webcamfee:
        success,img = cap.read()
    else:
        img = cv.imread("1.jpg")
    resize = cv.resize(img,(w,h))
    gray = cv.cvtColor(resize,cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray,(5,5),1)
    thres = utils1.valTrackbars()
    imthres = cv.Canny(blur,thres[0],thres[1]) 

    kernel = np.ones((5,5))
    imgdial = cv.dilate(imthres,kernel=kernel,iterations=2)
    imgthres = cv.erode(imgdial,kernel,iterations=2)

    # find contours
     
    imgcon = img.copy()
    imgbigcon = img.copy()

    contours, hierarchy = cv.findContours(imthres,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(imgcon, contours,-1,(0,222,222),10)
    


    #find the biggest contours

    biggest, maxarea = utils1.biggestContour(contours)
    if biggest.size!=0:
        biggest =utils1.reorder(biggest)
        cv.drawContours(imgbigcon,biggest,-1,(0,222,222),20)
        imgbigcon = utils1.drawRectangle(imgbigcon,biggest,2)
        pts1 = np.float32(biggest)

        pts2 = np.float32([[0,0],[w,0],[0,h],[w,h]])
        matrix = cv.getPerspectiveTransform(pts1,pts2)
        imgwarp = cv.warpPerspective(img,matrix,(w,h))


    #image 20 pixels removal

        imgwarp = imgwarp[20:imgwarp.shape[0]-20,20:imgwarp.shape[1]-20]
        imgwarp = cv.resize(imgwarp,(w,h))

    # apply adaptive threshhold to give it apaper like feel

        imgwarpgray = cv.cvtColor(imgwarp,cv.COLOR_BGR2GRAY)
        imgadap = cv.adaptiveThreshold(imgwarpgray,255,1,1,7,2)
        imgadap =cv.bitwise_not(imgadap)
        imgadap = cv.medianBlur(imgadap,3)

    #img array for display 

        imgarraya = ([img,gray,imgthres,imgcon],[imgbigcon,imgwarp,imgwarpgray,imgadap])

    else:

        imgarraya= ([img, gray, imgthres, imgcon], [
                    Blank, Blank, Blank, Blank])


    labels = [["orgiinal","gray","thresh","contours"],["biggestcon","warp","warpgray","adaptive"]]

    stackimg = utils1.stackImages(imgarraya,0.75,labels)

    cv.imshow("Result",stackimg)


    if cv.waitKey(1) & 0xFF == ord('s'):
        cv.imwrite("image/Yuvraj"+str(count)+".jpg", imgwarp)
        cv.rectangle(stackimg, ((int(stackimg.shape[1] / 2) - 230), int(stackimg.shape[0] / 2) + 50),
                      (1100, 350), (0, 255, 0), cv.FILLED)
        cv.putText(stackimg, "Scan Saved", (int(stackimg.shape[1] / 2) - 200, int(stackimg.shape[0] / 2)),
                    cv.FONT_HERSHEY_DUPLEX, 3, (0, 0, 255), 5, cv.LINE_AA)
        cv.imshow('Result', stackimg)
        cv.waitKey(300)
        count += 1






    
    

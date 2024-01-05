# Importing libraries
import cv2
import numpy as np
import khan_exten
from pytesseract import Output,pytesseract

########################################################################
khan_exten.initializeTrackbars()
count=0

while True:
    #Importing Image
    input_Img= cv2.imread(r"D:\Lectures\7th Semester\Robotics I\Project\3.jpeg")
    #Dimensions of image
    heightImg = input_Img.shape[0]
    widthImg = input_Img.shape[1]
    #Resize image to remove extra space
    input_Img = cv2.resize(input_Img, (widthImg, heightImg),interpolation = cv2.INTER_LINEAR) 
    # Blank Image created
    img_Blank = np.zeros((heightImg,widthImg, 3), np.uint8)
    # Image to grayscale
    img_Gray = cv2.cvtColor(input_Img, cv2.COLOR_BGR2GRAY)
    # Image with gaussianblur 
    img_Blur = cv2.GaussianBlur(img_Gray, (5, 5), 1)
    # Assigning trackbar
    thres=khan_exten.valTrackbars()
    # Canny for edges of image
    img_Threshold = cv2.Canny(img_Blur,thres[0],thres[1]) 
    kernel = np.ones((5, 5))
    # Dilating image
    imgDial = cv2.dilate(img_Threshold, kernel, iterations=2)
    # Erosion to image
    img_Threshold = cv2.erode(imgDial, kernel, iterations=1)  

    ## Detecting all contours
    #Copy image to contours
    imgContours = input_Img.copy()
    #Copy image to display biggest contour 
    imgBigContour = input_Img.copy() 
    #Locating all contours
    contours, hierarchy = cv2.findContours(img_Threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #Drawing detected contours
    cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 10)
    
    #Detecting the biggest contour
    biggest, maxArea = khan_exten.biggestContour(contours) 

    if biggest.size != 0:
        biggest=khan_exten.reorder(biggest)
        cv2.drawContours(imgBigContour, biggest, -1, (0, 255, 0), 20) # DRAW THE BIGGEST CONTOUR
        imgBigContour = khan_exten.drawRectangle(imgBigContour,biggest,2)
        #Creating points for the warping 
        pts1 = np.float32(biggest)
        pts2 = np.float32([[0, 0],[widthImg, 0], [0, heightImg],[widthImg, heightImg]])
        #Creating matrix to apply perspective warping
        Matrix = cv2.getPerspectiveTransform(pts1, pts2)
        img_Warp_Colored = cv2.warpPerspective(input_Img, Matrix, (widthImg, heightImg))

        #Removing pixels from the image for croppped image
        img_Warp_Colored=img_Warp_Colored[20:img_Warp_Colored.shape[0] - 20, 20:img_Warp_Colored.shape[1] - 20]
        img_Warp_Colored = cv2.resize(img_Warp_Colored,(widthImg,heightImg))

        #Applying adaptive threshold
        img_WarpGray = cv2.cvtColor(img_Warp_Colored,cv2.COLOR_BGR2GRAY)
        img_Adaptive= cv2.adaptiveThreshold(img_WarpGray, 255, 1, 1, 7, 2)
        img_Adaptive_2 = cv2.bitwise_not(img_Adaptive)
        img_Adaptive_3=cv2.medianBlur(img_Adaptive_2,3)
        
        #Extracting text from image
        array_text_stored = []
        text =pytesseract.image_to_string(img_Warp_Colored,lang="eng")
        array_text_stored.append(text)
        
        #Saving text into text file
        text_file = open(r"D:\Lectures\7th Semester\Robotics I\Project\text_my.txt", "w")
        n = text_file.write(text)
        text_file.close()

        #Image array for interface
        imageArray = ([input_Img,img_Gray,img_Threshold,imgContours],
                      [imgBigContour,img_Warp_Colored, img_WarpGray,img_Adaptive_3])

    else:
        imageArray = ([input_Img,img_Gray,img_Threshold,imgContours],
                      [img_Blank, img_Blank, img_Blank, img_Blank])

    #Display labels
    lables = [["Original","Gray","Threshold","Contours"],
              ["Biggest Contour","Warp Prespective","Warp Gray","Adaptive Threshold"]]

    stackedImage = khan_exten.stackImages(imageArray,0.4,lables)
    cv2.imshow("Result",stackedImage)
    

    # Creating a save option. Press S key
    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite("OCR\pro\scanned_mymage.jpeg", img_Warp_Colored)
        cv2.rectangle(stackedImage, ((int(stackedImage.shape[1] / 2) - 230), int(stackedImage.shape[0] / 2) + 50),
                      (1100, 350), (0, 255, 0), cv2.FILLED)
        cv2.putText(stackedImage, "Scan Saved", (int(stackedImage.shape[1] / 2) - 200, int(stackedImage.shape[0] / 2)),
                    cv2.FONT_HERSHEY_DUPLEX, 3, (0, 0, 255), 5, cv2.LINE_AA)
        cv2.imshow('Result', stackedImage)
        cv2.waitKey(300)
        count += 1

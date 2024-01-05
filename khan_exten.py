import cv2
import numpy as np

## TO STACK ALL THE IMAGES IN ONE WINDOW
def stackImages(img_Array,scale_Val,lables=[]):
    rows = len(img_Array)
    cols = len(img_Array[0])
    rowsAvailable = isinstance(img_Array[0], list)
    width = img_Array[0][0].shape[1]
    height = img_Array[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                img_Array[x][y] = cv2.resize(img_Array[x][y], (0, 0), None, scale_Val, scale_Val)
                if len(img_Array[x][y].shape) == 2: img_Array[x][y]= cv2.cvtColor( img_Array[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(img_Array[x])
            hor_con[x] = np.concatenate(img_Array[x])
        ver = np.vstack(hor)
        ver_con = np.concatenate(hor)
    else:
        for x in range(0, rows):
            img_Array[x] = cv2.resize(img_Array[x], (0, 0), None, scale_Val, scale_Val)
            if len(img_Array[x].shape) == 2: img_Array[x] = cv2.cvtColor(img_Array[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(img_Array)
        hor_con= np.concatenate(img_Array)
        ver = hor
    if len(lables) != 0:
        eachImgWidth= int(ver.shape[1] / cols)
        eachImgHeight = int(ver.shape[0] / rows)
        print(eachImgHeight)
        for d in range(0, rows):
            for c in range (0,cols):
                cv2.rectangle(ver,(c*eachImgWidth,eachImgHeight*d),(c*eachImgWidth+len(lables[d][c])*13+27,30+eachImgHeight*d),(255,255,255),cv2.FILLED)
                cv2.putText(ver,lables[d][c],(eachImgWidth*c+10,eachImgHeight*d+20),cv2.FONT_HERSHEY_COMPLEX,0.7,(255,0,255),2)
    return ver

def reorder(con_Points):

    con_Points = con_Points.reshape((4, 2))
    con_Points_New = np.zeros((4, 1, 2), dtype=np.int32)
    add = con_Points.sum(1)

    con_Points_New[0] = con_Points[np.argmin(add)]
    con_Points_New[3] =con_Points[np.argmax(add)]
    diff = np.diff(con_Points, axis=1)
    con_Points_New[1] =con_Points[np.argmin(diff)]
    con_Points_New[2] = con_Points[np.argmax(diff)]

    return con_Points_New


def biggestContour(contours):
    biggest = np.array([])
    Max_Area = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area > 5000:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)
            if area > Max_Area and len(approx) == 4:
                biggest = approx
                Max_Area = area
    return biggest,Max_Area

def drawRectangle(input_Img,biggest,thickness_value):
    cv2.line(input_Img, (biggest[0][0][0], biggest[0][0][1]), (biggest[1][0][0], biggest[1][0][1]), (0, 255, 0), thickness_value)
    cv2.line(input_Img, (biggest[0][0][0], biggest[0][0][1]), (biggest[2][0][0], biggest[2][0][1]), (0, 255, 0), thickness_value)
    cv2.line(input_Img, (biggest[3][0][0], biggest[3][0][1]), (biggest[2][0][0], biggest[2][0][1]), (0, 255, 0), thickness_value)
    cv2.line(input_Img, (biggest[3][0][0], biggest[3][0][1]), (biggest[1][0][0], biggest[1][0][1]), (0, 255, 0), thickness_value)

    return input_Img

def nothing(x):
    pass

def initializeTrackbars(intialTracbarVals=0):
    cv2.namedWindow("Trackbars")
    cv2.resizeWindow("Trackbars", 360, 240)
    cv2.createTrackbar("Threshold1", "Trackbars", 100,255, nothing)
    cv2.createTrackbar("Threshold2", "Trackbars", 100, 255, nothing)


def valTrackbars():
    Threshold1 = cv2.getTrackbarPos("Threshold1", "Trackbars")
    Threshold2 = cv2.getTrackbarPos("Threshold2", "Trackbars")
    src = Threshold1,Threshold2
    return src
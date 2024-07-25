import cv2 as cv

#testing camera
width,height=1280,720

cap=cv.VideoCapture(1)

cap.set(3,width)
cap.set(4,height)

while True:
    success,img= cap.read()
    cv.imshow("Image",img)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
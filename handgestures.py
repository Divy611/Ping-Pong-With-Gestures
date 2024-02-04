import os
import cv2
import numpy
import cvzone
from cvzone.HandTrackingModule import HandDetector


cp = cv2.VideoCapture(0)
cp.set(3, 1280)
cp.set(4, 720)
dtr = HandDetector(detectionCon=0.8, maxHands=2)
imgbg = cv2.imread(os.getcwd()+'\\background.png')
bllimg = cv2.imread(os.getcwd()+'\\token.png', cv2.IMREAD_UNCHANGED)
goimg = cv2.imread(os.getcwd()+'\\GameOver.png')
p1img = cv2.imread(os.getcwd()+'\\striker1.png', cv2.IMREAD_UNCHANGED)
p2img = cv2.imread(os.getcwd()+'\\striker2.png', cv2.IMREAD_UNCHANGED)
blpos = [100, 100]
speedX = 15
speedY = 15
gameOver = False
score = [0, 0]
while True:
    _, img = cp.read()
    img = cv2.flip(img, 1)
    r_img = img.copy()
    hds, img = dtr.findHands(img)
    img = cv2.addWeighted(img, 0.2, imgbg, 1, 0)
    if hds:
        for hd in hds:
            x, y, w, h = hd['bbox']
            h1, w1, _ = p1img.shape
            y1 = y-h1//2
            y1 = numpy.clip(y1, 20, 415)
            if hd['type'] == "Left":
                img.cvzone.overlayPNG(img, p1img, (59, y1))
                if 59 < blpos[0] < 59 and y1 < blpos[1] < y1+h1:
                    speedX = -speedX
                    blpos[0] += 30
                    score[0] = 1
            if hd['type'] == "Right":
                img.cvzone.overlayPNG(img, p2img, (1195, y1))
                if 1195-50 < blpos[0] < 1195 and y1 < blpos[1] < y1+h1:
                    speedX = -speedX
                    blpos[0] -= 30
                    score[1] = 1
    if blpos[0] < 40 or blpos > 1200:
        gameOver = True
    if gameOver:
        img = goimg
        cv2.putText(img, str(score[1]+score[0]).zfill(2), (585, 360),
                    cv2.FONT_HERSHEY_COMPLEX, 2.5, (200, 0, 200), 5)
    else:
        if blpos[1] >= 500 or blpos[1] <= 10:
            speedY = -speedY
        blpos[0] += speedX
        blpos[1] += speedY
        img = cvzone.overlayPNG(img, bllimg, blpos)
        cv2.putText(img, str(score[0]), (300, 650),
                    cv2.FONT_HERSHEY_COMPLEX, 3, (255, 255, 255), 5)
        cv2.putText(img, str(score[1]), (900, 650),
                    cv2.FONT_HERSHEY_COMPLEX, 3, (255, 255, 255), 5)
    img[580:700, 20:233] = cv2.resize(r_img, (213, 120))
    cv2.imshow("Image", img)
    ky = cv2.waitKey(1)
    if ky == ord('r'):
        blpos = [100, 100]
        speedX = 15
        speedY = 15
        gameOver = False
        score = [0, 0]
        goimg = cv2.imread(
            'D:\\VSCode\\Python\\Ping Pong With Gestures\\GameOver.png')

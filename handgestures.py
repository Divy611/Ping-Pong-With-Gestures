import os
import cv2
import numpy as np
import cvzone
from cvzone.HandTrackingModule import HandDetector

cp = cv2.VideoCapture(0)
cp.set(3, 1280)
cp.set(4, 720)
dtr = HandDetector(detectionCon=0.8, maxHands=2)
imgbg = cv2.imread(os.getcwd()+'\\images\\background.png')
bllimg = cv2.imread(os.getcwd()+'\\images\\token.png', cv2.IMREAD_UNCHANGED)
goimg = cv2.imread(os.getcwd()+'\\images\\GameOver.png')
p1img = cv2.imread(os.getcwd()+'\\images\\striker1.png', cv2.IMREAD_UNCHANGED)
p2img = cv2.imread(os.getcwd()+'\\images\\striker2.png', cv2.IMREAD_UNCHANGED)
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
    if img.shape != imgbg.shape:
        img = cv2.resize(img, (imgbg.shape[1], imgbg.shape[0]))

    if img.shape[2] != imgbg.shape[2]:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if img.dtype != imgbg.dtype:
        img = img.astype(imgbg.dtype)
    img = cv2.addWeighted(img, 0.2, imgbg, 1, 0)

    if hds:
        for hd in hds:
            x, y, w, h = hd['bbox']
            h1, w1, _ = p1img.shape
            y1 = y - h1 // 2
            y1 = np.clip(y1, 20, 415)

            if hd['type'] == "Left":
                img = cvzone.overlayPNG(img, p1img, (59, y1))
                if 59 < blpos[0] < 59 + w1 and y1 < blpos[1] < y1 + h1:
                    speedX = -speedX
                    blpos[0] += 30
                    score[0] = 1

            if hd['type'] == "Right":
                img = cvzone.overlayPNG(img, p2img, (1195, y1))
                if 1195 - 50 < blpos[0] < 1195 and y1 < blpos[1] < y1 + h1:
                    speedX = -speedX
                    blpos[0] -= 30
                    score[1] = 1

    if blpos[0] < 40 or blpos[0] > 1200:
        gameOver = True

    if gameOver:
        img = goimg
        cv2.putText(img, str(score[1] + score[0]).zfill(2), (585, 360),
                    cv2.FONT_HERSHEY_COMPLEX, 2.5, (200, 0, 200), 5)
        ky = cv2.waitKey(1)
        if ky == ord('r'):
            blpos = [100, 100]
            speedX = 15
            speedY = 15
            gameOver = False
            score = [0, 0]

    else:
        if blpos[1] >= 500 or blpos[1] <= 10:
            speedY = -speedY
        blpos[0] += speedX
        blpos[1] += speedY

        if bllimg.shape[2] == 4:
            img = cvzone.overlayPNG(img, bllimg, blpos)
        else:
            print("")
        resized_r_img = cv2.resize(r_img, (213, 120))
        h, w, _ = img.shape
        start_h = max(0, h - 120)
        r_img = img[start_h:start_h+120, 20:233].copy()
        # img[580:700, 20:233] = resized_r_img

        # r_img = img[580:700, 20:233].copy()
        # r_img_resized = cv2.resize(r_img, (213, 120))
        # r_img_resized = cv2.cvtColor(r_img_resized, cv2.COLOR_BGR2RGB)
        # img[580:700, 20:233] = r_img_resized

        cv2.putText(img, str(score[0]), (300, 650),
                    cv2.FONT_HERSHEY_COMPLEX, 3, (255, 255, 255), 5)
        cv2.putText(img, str(score[1]), (900, 650),
                    cv2.FONT_HERSHEY_COMPLEX, 3, (255, 255, 255), 5)

    cv2.imshow("Image", img)
    ky = cv2.waitKey(1)
    if ky == ord('r'):
        blpos = [100, 100]
        speedX = 15
        speedY = 15
        gameOver = False
        score = [0, 0]

    if ky == 27:
        break

cv2.destroyAllWindows()

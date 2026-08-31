"""
tools/calibrate_color.py - HSV range tuner for color tracker
Usage: python tools/calibrate_color.py
Sliders to tune HSV bounds for your glove/marker.
Press 's' to save to config.py range, 'q' to quit.
"""
import cv2
import numpy as np

def nothing(x): pass

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

cv2.namedWindow("Calibrate", cv2.WINDOW_NORMAL)
cv2.namedWindow("Mask", cv2.WINDOW_NORMAL)
cv2.createTrackbar("H_low", "Calibrate", 40, 179, nothing)
cv2.createTrackbar("S_low", "Calibrate", 70, 255, nothing)
cv2.createTrackbar("V_low", "Calibrate", 70, 255, nothing)
cv2.createTrackbar("H_high", "Calibrate", 80, 179, nothing)
cv2.createTrackbar("S_high", "Calibrate", 255, 255, nothing)
cv2.createTrackbar("V_high", "Calibrate", 255, 255, nothing)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))

print("Tune sliders to isolate your marker. Press 's' to print values, 'q' to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        continue
    frame = cv2.flip(frame, 1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hl = cv2.getTrackbarPos("H_low", "Calibrate")
    sl = cv2.getTrackbarPos("S_low", "Calibrate")
    vl = cv2.getTrackbarPos("V_low", "Calibrate")
    hh = cv2.getTrackbarPos("H_high", "Calibrate")
    sh = cv2.getTrackbarPos("S_high", "Calibrate")
    vh = cv2.getTrackbarPos("V_high", "Calibrate")
    lower = np.array([hl, sl, vl])
    upper = np.array([hh, sh, vh])
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    # Find and draw centroid
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) > 400:
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx = int(M["m10"]/M["m00"]); cy = int(M["m01"]/M["m00"])
                cv2.circle(frame, (cx,cy), 8, (0,255,0), -1)
                cv2.drawContours(frame, [c], -1, (255,0,0), 2)
    cv2.imshow("Calibrate", frame)
    cv2.imshow("Mask", mask)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('q') or k == 27:
        break
    elif k == ord('s'):
        print(f"HSV_LOWER = ({hl}, {sl}, {vl})")
        print(f"HSV_UPPER = ({hh}, {sh}, {vh})")
        print("Copy these into config.py")

cap.release()
cv2.destroyAllWindows()

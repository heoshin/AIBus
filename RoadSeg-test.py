import cv2
import numpy as np
from utils.opv import OpvModel
import matplotlib.pyplot as plt
import time

import serial

ser = serial.Serial('COM5', 115200)

def sendSerial(cmd) :
    sendStr = str(cmd)
    global ser
    sendStr = sendStr + '\n'
    sendStr = sendStr.encode('utf-8')
    ser.write(sendStr)

def servoControl(distance) :
    moter = 60 / 100 * distance
    servoVal = int(100 - moter)
    sendSerial(servoVal)

def speedControl(speedVal) :
    sendSerial(speedVal)

def controlAICar(distance, speed):
    servoControl(distance * 1.5)
    speedControl(speed)

Dw_ms = time.time()
cnt = 0
isPause = True
stan_X = 500
stan_Y = 512 - 150

class_names = ['BG', 'road', 'curb', 'mark']

np.set_printoptions(linewidth=np.inf)
mymodel0 = OpvModel("road-segmentation-adas-0001", device="CPU", fp="FP32")

# cam = cv2.VideoCapture('https://10.129.204.235:8080/video')
cam = cv2.VideoCapture('./road/1_1.mp4')

rightSlope, leftSlope, rightIntercept, leftIntercept = [],[],[],[]
def draw_lines(img, lines, thickness=5):
    global rightSlope, leftSlope, rightIntercept, leftIntercept
    rightColor=[0,255,0]
    leftColor=[255,0,0]

    for line in lines:
        for x1,y1,x2,y2 in line:
            slope = (y1-y2)/(x1-x2)
            if slope > 0.3:
                if x1 > 500 :
                    yintercept = y2 - (slope*x2)
                    rightSlope.append(slope)
                    rightIntercept.append(yintercept)
                else: None
            elif slope < -0.3:
                if x1 < 600:
                    yintercept = y2 - (slope*x2)
                    leftSlope.append(slope)
                    leftIntercept.append(yintercept)
                    
    leftavgSlope = np.mean(leftSlope[-30:])
    leftavgIntercept = np.mean(leftIntercept[-30:])
    rightavgSlope = np.mean(rightSlope[-30:])
    rightavgIntercept = np.mean(rightIntercept[-30:])
    
    try:
        left_line_x1 = int((0.65*img.shape[0] - leftavgIntercept)/leftavgSlope)
        left_line_x2 = int((img.shape[0] - leftavgIntercept)/leftavgSlope)
        right_line_x1 = int((0.65*img.shape[0] - rightavgIntercept)/rightavgSlope)
        right_line_x2 = int((img.shape[0] - rightavgIntercept)/rightavgSlope)
        pts = np.array([[left_line_x1, int(0.65*img.shape[0])],[left_line_x2, int(img.shape[0])],[right_line_x2, int(img.shape[0])],[right_line_x1, int(0.65*img.shape[0])]], np.int32)
        pts = pts.reshape((-1,1,2))
        cv2.fillPoly(img,[pts],(0,50,0))
        left_line = [(left_line_x1, int(0.65*img.shape[0])), (left_line_x2, int(img.shape[0]))]
        right_line = [(right_line_x1, int(0.65*img.shape[0])), (right_line_x2, int(img.shape[0]))]
        cv2.line(img, left_line[0], left_line[1], leftColor, 3)
        cv2.line(img, right_line[0], right_line[1], rightColor, 3)
        return left_line, right_line
    except ValueError:
        pass
    return [(0, 0), (0, 0)], [(0, 0), (0, 0)]

def linearPoint(x1, y1, x2, y2, pointY):
    # x1, y1, x2, y2 = fir[0], fir[1], sec[0], sec[1]
    # x = ((x2 - x1) / (y2 - y1)) * (pointY - y1) + x1
    yPlus = y2 - y1
    xPlus = x2 - x1
    if yPlus == 0:
        yPlus = 1
    if xPlus == 0:
        xPlus = 1

    m = yPlus / xPlus
    b = y1 - m * x1

    x = (pointY - b) / m
    return (int(x), 512 - pointY)

while True:
    #frame 별로 복사
    ret, inputFrame = cam.read()
    frame = cv2.resize(inputFrame, dsize=(896, 512), interpolation=cv2.INTER_AREA)
    #모델에 입력 후 예측값 저장
    predictions = mymodel0.Predict(frame)
    img_Gray = predictions['L0317_ReWeight_SoftMax'][0]
    #frame 복사
    all_frame = np.zeros((512, 896, 3), dtype=np.uint8)
    all_frame = frame
    #각 요소들 합치기
    image12 = cv2.hconcat([img_Gray[0], img_Gray[1]])
    image34 = cv2.hconcat([img_Gray[2], img_Gray[3]])
    allImg_Gray = cv2.vconcat([image12, image34])
    #mark 이진화
    ret, thresh = cv2.threshold((img_Gray[3]) * 255, 200, 255, cv2.THRESH_BINARY)
    #허프라인 생성
    lines = cv2.HoughLinesP(thresh.astype(np.uint8), 1, np.pi/180, 30, maxLineGap=200)
    #허프라인들의 프레임 생성
    houghline = thresh.copy()
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(houghline, (x1, y1), (x2, y2), (255, 0, 0), 3)
    #평균값 계산으로 두 개의 직선으로 치환
    houghAve = np.zeros((512, 896, 3), dtype=np.uint8)
    left_line, right_line = draw_lines(houghAve, lines)
    #직선의 기준점 지정
    left_Point = linearPoint(left_line[0][0], 512 - left_line[0][1],left_line[1][0], 512 - left_line[1][1], 150)
    right_Point = linearPoint(right_line[0][0], 512 - right_line[0][1], right_line[1][0], 512 - right_line[1][1], 150)
    center_Point = (int((left_Point[0] + right_Point[0]) / 2), 512 - 150)
    #원본과 직선 합치기
    all_frame[thresh == 255] = [0, 0, 255]
    all_frame = cv2.add(all_frame, houghAve)
    #기준점 그리기
    cv2.circle(all_frame, left_Point, 5, (0,255,0), -1)
    cv2.circle(all_frame, right_Point, 5, (0,255,0), -1)
    cv2.circle(all_frame, center_Point, 5, (0,255,0), -1)
    #기준선 그리기
    cv2.line(all_frame, (stan_X, 512), (stan_X, stan_Y), (0, 0, 255), 1)
    cv2.line(all_frame, (stan_X, stan_Y), center_Point, (0, 0, 255), 1)
    #기준점을 기반으로 AICar 컨트롤
    controlAICar(center_Point[0] - stan_X, 400)
    
    cv2.putText(all_frame, str(round((time.time() - Dw_ms) * 1000, 2)) + 'ms', (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('image', allImg_Gray)
    cv2.imshow('thresh', thresh)
    cv2.imshow('houghLine', houghline)
    cv2.imshow('img_line', houghAve)
    cv2.imshow("all_frame", all_frame)

    Key = cv2.waitKey(1) & 0xFF
    if True:
        if Key == ord('q'):
            controlAICar(stan_X, 0)
            break
        elif Key == ord('p'):
            isPause = not isPause
        elif isPause:
            controlAICar(stan_X, 0)
            while True:
                Key = cv2.waitKey(1) & 0xFF
                if Key == ord('p'):
                    isPause = False
                    break
                elif Key == ord(' '):
                    break

    Dw_ms = time.time()

cam.release()
cv2.destroyAllWindows()
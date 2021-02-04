import cv2
import numpy as np
import os
import io
import time
import multiprocessing
from csi_camera import CSI_Camera
import copy

low_threshold = 0
high_threshold = 150
rho = 1                 # distance resolution in pixels of the Hough grid
theta = np.pi / 180     # angular resolution in radians of the Hough grid
threshold = 200         # minimum number of votes (intersections in Hough grid cell)
max_line_gap = 20 

class Yolo_Line() :
    def __init__(self) :
        self.net = cv2.dnn.readNet("yolov3-tiny_3000.weights", "yolov3-tiny.cfg")    # 학습 모델을 불러옴
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        self.classes = ["back", "front"]
        self.layer_names = self.net.getLayerNames()
        self.output_layers = [self.layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
    
    def detect_lines(self, img) :
        blob = cv2.dnn.blobFromImage(img, 0.00392, (448, 448), (104.0,177.0, 123.0), True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)

        confidences_back = []
        confidences_front = []
        boxes_back = []
        boxes_front = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.3 :
                    # Object detected
                    center_x = int(detection[0] * 720)
                    center_y = int(detection[1] * 720)
                    w = int(detection[2] * 720)
                    h = int(detection[3] * 720)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    if class_id :
                        boxes_front.append([x, y, w, h, class_id])
                        confidences_front.append(float(confidence))
                    else :
                        boxes_back.append([x, y, w, h, class_id])
                        confidences_back.append(float(confidence))

        cv2.dnn.NMSBoxes(boxes_front, confidences_front, 0.3, 0.2)
        cv2.dnn.NMSBoxes(boxes_back, confidences_back, 0.3, 0.2)
        return boxes_back, boxes_front

def get_interest(img, k) : # 라이터 위치를 찾기 위한 이미지의 절반을 흑백 처리 함수
    if k :
        img[0:360, :] = 0
    else :
        img[360:, :] = 0
    return img

def checkHeadRatio(raw, stick) :    # 라이터 헤드 사이 간격을 알기 위한 좌표 추정 함수
    #return int((stick-raw) * (5/105))
    return int(((stick-raw) * (50/205))/2)

def checkBetweenRatio(raw, stick) :    # 라이터 헤드 사이 간격을 알기 위한 좌표 추정 함수
    return int((stick-raw) * (30/105))

def checkLineRatio(stick, raw) :
    return raw + int((stick-raw) * (12/40)), raw + int((stick-raw) * (19.5/40)), raw + int((stick-raw) * (32/40))
    #return raw + int((stick-raw) * (12/39)), raw + int((stick-raw) * (22/39)), raw + int((stick-raw) * (32/39))

def findRaw(img) :  # 라이터 고정대 좌표를 찾기 위한 함수
    result = []

    for k in range(2) :     # 1 : 트레이 찾기, 0 : 상위 기준선 찾기
        gray = get_interest(copy.deepcopy(img), k)
        kernel_size = 5

        for i in range(1) :     # 여기 시작 전에 수정
            gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)

        edges = cv2.Canny(gray, low_threshold, high_threshold)
        min_line_length = 0  # minimum number of pixels making up a line

        lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)

        candidate = []
        if lines is not None :
            for line in lines:
                for x1,y1,x2,y2 in line:
                    if 355*(1+k)> y1 > 365*k :
                        candidate.append([y1, y2])

        if candidate :
            candidate.sort(key = lambda x : x[0])
            if k : result.append(candidate[0][0])
            else : result.append(candidate[0][0]+8)

        else :
            result.append(-1)
    return result[0], result[1]

def getCapture() :   # 반복적으로 화면 캡쳐를 얻는 함수
    # 로컬에 화면 캡쳐 이미지를 저장함
    camera = CSI_Camera()
    camera.create_gstreamer_pipeline(
        sensor_id = 0,
        sensor_mode = 0,
        framerate = 30,
        flip_method = 0,
        display_height = 720,
        display_width = 1280
    )
    camera.open(camera.gstreamer_pipeline)
    camera.start()
    
    try:
        # camera.start_counting_fps()
        while 1:
            _, img_ori = camera.read()
            
            
            temp = cv2.cvtColor(img_ori, cv2.COLOR_RGB2GRAY)
            temp = cv2.cuda.cvtColor(img_ori, cv.COLOR_RGB2GRAY)
            
            
            
            img = np.zeros([720, 720, 3])
            for i in range(3) :
               img[:,:,i] = temp[:,280:1000]
               
            img = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5, scale=0.15)
            img = cv2.cuda.createSobelFilter(cv.CV_8UC1, -1, 1, 1)
            
            
            img = cv2.convertScaleAbs(img)
            
            sub_img = copy.deepcopy(img)
            
            print("1")
            raw=0
            try :
                print("2")
                temp, stick = findRaw(img)      # 라이터 위치를 특정하기 위한 받침대 위치 확인
                print(temp,stick)
                if temp is not None and temp > 0 :
                    # if 0.9*raw < temp < 1.1*raw : print("카메라가 위치를 벗어남")   # 이전 프레임과 비교하여 받침대 위치가 벗어나면 카메라가 움직인 것
                    raw = temp
                
                print("3")
                # 상한~하한선 찾기
                print(stick, raw)
                h_line, m_line, l_line = checkLineRatio(stick, raw)
                
                print(h_line, m_line, l_line)
                
                cv2.line(sub_img, (0, raw), (720, raw), (255,255,0), 1)
                cv2.line(sub_img, (0, stick), (720, stick), (255,255,0), 1)
                cv2.line(sub_img, (0, h_line), (720, h_line), (0,255,0), 1)
                cv2.line(sub_img, (0, m_line), (720, m_line), (0,255,0), 1)
                cv2.line(sub_img, (0, l_line), (720, l_line), (0,255,0), 1)
            
                cv2.imwrite("images/"+str(10)+".jpg", sub_img)
                
            except:
                pass
            
            
            cv2.imshow("Gas Solution", sub_img)
            time.sleep(10)
            
            
            
            
            

            
            
            if (cv2.waitKey(5) & 0xFF) == 27: break
    finally:
        camera.stop()
        camera.release()
        cv2.destroyAllWindows()
        
getCapture()
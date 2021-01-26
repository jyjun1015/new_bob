import cv2
import numpy as np
import os
import io
import time
import multiprocessing
import copy
from csi_camera import CSI_Camera



net = cv2.dnn.readNet("yolov3-tiny_4000.weights", "yolov3-tiny.cfg")    # 학습 모델을 불러옴
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

raw = 274 # 라이터 고정대 길이 
stick = 490


stickers=[]
stickers_error=[]
for i in range(12) :    # 스티커 불량 여부를 판정하기 위한(템플릿 매칭에 사용될) 기준 스티커를 불러옴
    if os.path.isfile('nnum'+str(i)+'.JPG') :
        stickers.append(cv2.imread('nnum'+str(i)+'.JPG'))
for k in range(1,4) :     # 에러 템플릿 꼭 있어야 하는가? ->
    if os.path.isfile('error'+str(k)+'.JPG') :
        stickers_error.append(cv2.imread('error'+str(k)+'.JPG'))

def checkHeadRatio(raw, stick) :    # 라이터 헤드를 찾기 위한 좌표 추정 함수
    return int((stick-raw) * (5/105))    #헤드 간 간격/몸통 길이

def checkStickRatio(raw) :
    return int(raw * (490/274))

def checkStickerRatio(raw, stick) :
    return int((stick-raw)*(45/216)), int((stick-raw)*(133/216))

def checkStickerRatio2(raw, stick) :
    return int((stick-raw)*(134/216)), int((stick-raw)*(175/216))

def getCapture() :   # 반복적으로 화면 캡쳐를 얻는 함수
    # 로컬에 화면 캡쳐 이미지를 저장함
    camera = CSI_Camera()
    camera.create_gstreamer_pipeline(
        sensor_id = 0,
        sensor_mode = 0,
        framerate = 30,
        flip_method = 2,
        display_height = 720,
        display_width = 1280
    )
    camera.open(camera.gstreamer_pipeline)
    camera.start()
    # cv2.namedWindow("Sticker Solution", cv2.WINDOW_AUTOSIZE)
    
    if not camera.video_capture.isOpened():
        print("Unable to open camera")
        SystemExit(0)
    
    try:
        camera.start_counting_fps()
        # while cv2.getWindowProperty("Sticker Solution", 0) >= 0:
        while True:
            _, img = camera.read()
            img = img[:,280:1000,:]

            # Create a 4D blolb from a frame
            blob = cv2.dnn.blobFromImage(img, 0.00392, (448, 448), (0, 0, 0), True, crop=False)
            
            # Sets the input to the network
            net.setInput(blob)
            
            # Run the forward pass to get output of the output layers
            outs = net.forward(output_layers)
            boxes = []
            boxes_low = -1
            
            # each detection from each ouput layer
            # for out in outs:
            for detection in outs[1]:

                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
 
                if confidence > 0.5 and class_id == 0 :
    
                    # Object detected
                    center_x = int(detection[0] * 720)
                    center_y = int(detection[1] * 720)
                    w = int(detection[2] * 720)
                    h = int(detection[3] * 720)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,255), 2)
                    # cv2.imshow("Sticker Solution", img)
                    boxes.append([x, y, w, h])
                    boxes_low = max(boxes_low, y+h)
                    
            if len(boxes) < 7 :
                cv2.imshow("Sticker Solution", img)
                continue
                
                # camera.frames_displayed += 1
                # cap = cap + 1
            print("라이터 헤드 검출",len(boxes))
            
            try:
                
                boxes.sort(key=lambda x : x[0])
                temp = boxes_low
                temp2 = checkStickRatio(temp)
                if temp2 > 0 :
                    # if 0.9*stick < temp2 < 1.1*stick :
                    stick = temp2
                if temp > 0 :
                    # if 0.9*raw < temp < 1.1*raw : 
                    raw = temp
                between = checkHeadRatio(raw, stick)
               
                error_region = []
                error_minVal = 1
                start = boxes[0][0]-between if boxes[0][0]-between >= 0 else 0
                end = boxes[-1][0]+boxes[-1][2] + between if boxes[-1][0]+boxes[-1][2] + between < 720 else 719
                cut_img = img[raw:stick, start:end]
            
            
                for error in stickers_error:
                    error_temp = copy.deepcopy(error)
                    # 에러 상황 저장
                    if stick-raw <= error.shape[0] : # 가로길이가 사진의 가로보다 작으면 
                        resize_width, resize_height = checkStickerRatio2(raw, stick)
                        error_temp = cv2.resize(error, (resize_width, resize_height), interpolation=cv2.INTER_LINEAR)
                    result = cv2.matchTemplate(cut_img, error_temp, cv2.TM_SQDIFF_NORMED)
                    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(result)
                    x, y = minLoc
                    h, w, c = error_temp.shape

                    if minVal < 0.09 and error_minVal > minVal :
                        print("exits")
                        
                        if error_region : error_region[0] = [start+x, raw+y, w, h, minVal]
                        else : error_region.append([start+x, raw+y, w, h, minVal])
                
                if error_region :
                    cv2.rectangle(img, (error_region[0][0], error_region[0][1]), (error_region[0][0]+error_region[0][2], error_region[0][1]+error_region[0][2]), (0, 255, 255), 2, cv2.LINE_8)
                print("error_region :", error_region)

                results = []
                
                for index in boxes :
                    start = index[0]-between if index[0]-between >= 0 else 0
                    end = index[0]+index[2]+between if index[0]+index[2]+between < 720 else 719 #여기 사이즈 바꿔야함
                    resize_width, resize_height = checkStickerRatio(raw, stick)

                    cut_img = img[raw:stick, start:end]
                    # cv2.rectangle(sub_img, (start,raw), (end, stick), (255, 255, 0), 2, cv2.LINE_8)

                    resul = []
                    for sticker in stickers :
                        sticker_temp = copy.deepcopy(sticker)
                        
                        #-----이거 나중에 높이 대비 스티커 길이 비율로 수정하기-----#
                        sticker_temp = cv2.resize(sticker, (resize_width, resize_height), interpolation=cv2.INTER_LINEAR)
                        result = cv2.matchTemplate(cut_img, sticker_temp, cv2.TM_SQDIFF_NORMED)
                        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(result)
                        x, y = minLoc
                        h, w, c = sticker_temp.shape
                        resul.append([start+x, raw+y, w, h, minVal])
                    resul.sort(key = lambda x : x[4])
                    
                    print('resul :',len(resul), resul[0][4])
                
                    if resul[0][4] <= 0.10 :
                        if error_region :
                            if error_region[0][0] < resul[0][0]+resul[0][2] < error_region[0][0]+error_region[0][2] or error_region[0][0] < resul[0][0] < error_region[0][0]+error_region[0][2] :
                                continue
                        results.append(resul[0])
                        
                print("result :",len(results))
                

                for i, index in enumerate(results) :
                    cv2.rectangle(img, (index[0], index[1]), (index[0]+index[2], index[1]+index[3]), (255, 255, 0), 2, cv2.LINE_8)

                print("최종 스티커 개수",len(results))
                
            except: pass
            
            cv2.imshow("Sticker Solution", img)
            print()
            if (cv2.waitKey(5) & 0xFF) == 27: break
    finally:
        camera.stop()
        camera.release()
        cv2.destroyAllWindows()


getCapture()


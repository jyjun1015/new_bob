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

stickers=[]

for i in range(3,4,1) :    # 스티커 불량 여부를 판정하기 위한(템플릿 매칭에 사용될) 기준 스티커를 불러옴
    stickers.append(cv2.imread("./new_template/"+str(i)+'.jpg'))

sticker = stickers[0]

def checkHeadRatio(raw, stick) :    # 라이터 헤드를 찾기 위한 좌표 추정 함수
    return int((stick-raw) * (5/105))    #헤드 간 간격/몸통 길이

def checkStickRatio(raw) :
    return int(raw * (490/260))

def checkStickerRatio(raw, stick) :
    return int((stick-raw)*(45/216)), int((stick-raw)*(133/216))

def checkStickerRatio2(raw, stick) :
    return int((stick-raw)*(134/216)), int((stick-raw)*(175/216))

  
# 로컬에 화면 캡쳐 이미지를 저장함
camera = CSI_Camera()
camera.create_gstreamer_pipeline(
    sensor_id = 0,
    sensor_mode = 1,
    framerate = 30,
    flip_method = 2,
    display_height = 720,
    display_width = 1280
)
camera.open(camera.gstreamer_pipeline)
camera.start()
# cv2.namedWindow("Sticker Solution", cv2.WINDOW_AUTOSIZE)

    
    
def getStart():
    try:
        # whats this
        # camera.start_counting_fps()
        # while cv2.getWindowProperty("Sticker Solution", 0) >= 0:
        while True:
         
            _, img = camera.read()
            img = img[:,280:1000,:]

            # sub_img = copy.deepcopy(img)
            if (cv2.waitKey(5) & 0xFF) == 13:
                print("Start!!!!!")
                res = []
                for _ in range(3):
                
                    try :
                        blob = cv2.dnn.blobFromImage(img, 0.00392, (448, 448), (0, 0, 0), True, crop=False)
                        net.setInput(blob)
                        outs = net.forward(output_layers)
                        boxes_temp = []
                        confidences = []
                        boxes_low = -1
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
                                confidences.append(float(confidence))
                                #cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,255), 2)
                                boxes_temp.append([x, y, w, h])
                        
                                boxes_low = max(boxes_low, y+h+20)
                                  
                        num = len(boxes_temp)
             
                        indexes = cv2.dnn.NMSBoxes(boxes_temp, confidences, 0.3, 0.2)
                      
                        boxes = []
              
                        for i in range(num):
                  
                            if i in indexes:
                   
                                x, y, w, h = boxes_temp[i]
                                boxes.append([x,y,w,h])
                                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
                                
                        num2 = len(boxes)
                        
                        if num < 5 :
                            print("라이터 헤드 수 부족",num2)
                            #cv2.imshow("Sticker Solution", img)
                            #continue;
                        
                        print("라이터 헤드 검출", num2)
                        
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

                        start = boxes[0][0]-between if boxes[0][0]-between >= 0 else 0
                        end = boxes[-1][0]+boxes[-1][2] + between if boxes[-1][0]+boxes[-1][2] + between < 720 else 719
                        cut_img = img[raw:stick, start:end]
                        cv2.imwrite("3.jpg", cut_img)
                        time.sleep(1)
                        
                  
                        results = []
                    
                    # 검출한 것에 대하여...판단 
                        for index in boxes :
                            start = index[0]-between+20 if index[0]-between >= 0 else 0
                            end = index[0]+index[2]+between-8 if index[0]+index[2]+between < 720 else 719 #여기 사이즈 바꿔야함
                            resize_width, resize_height = checkStickerRatio(raw, stick)
                            cut_img = img[raw:stick, start:end]

                            cut_img = cv2.medianBlur(cut_img,5)

                            resul = []
                            # for sticker in stickers :
                                                    #-----이거 나중에 높이 대비 스티커 길이 비율로 수정하기-----#
                                # 사이즈 변경 왜 하는지? 필요 없음 
                               
                            sticker_temp = cv2.resize(sticker, (int(sticker.shape[1]*2/3), int(sticker.shape[0]*2/3)), interpolation=cv2.INTER_LINEAR)
                            try:
                                result = cv2.matchTemplate(cut_img, sticker_temp, cv2.TM_SQDIFF_NORMED)
                            except:
                                break;
                            minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(result)
                            print(minVal)
                            x, y = minLoc
                            h, w, c = sticker_temp.shape
                    
                            resul.append([start+x, raw+y, w, h, minVal])
                     

                            # resul.sort(key = lambda x : x[4])
                            if resul[0][4] <= 0.15 :
                            
                                results.append(resul[0])
                           
                        print("result :",len(results))

                        for i, index in enumerate(results) :
                            cv2.rectangle(img, (index[0], index[1]), (index[0]+index[2], index[1]+index[3]), (255, 255, 0), 2, cv2.LINE_8)

                        if len(results) < 10 :
             
                            res.append("False")
                            
                        if len(results) == 10:
                            res.append("True")
                 
                                    
                        for i, index in enumerate(results) :
                            cv2.rectangle(img, (index[0], index[1]), (index[0]+index[2], index[1]+index[3]), (255, 255, 0), 2, cv2.LINE_8)

                        cv2.imshow("Sticker Solution", img)
                        if (cv2.waitKey(5) & 0xFF) == 27: break
                    except:
                        cv2.imshow("Sticker Solution", img)
                        if (cv2.waitKey(5) & 0xFF) == 27: break
                        
                    print("result : ",res)
                    
                if "True" in res:
                    print("PASS")
                else:
                    print("FAIL")
                
                    
            elif (cv2.waitKey(5) & 0xFF) == 27: break
            
            else : cv2.imshow("Sticker Solution", img)
            
    except:
        print("END")
        pass
    finally:
        camera.stop()
        camera.release()
        cv2.destroyAllWindows()



# button start -> 5 frame calc
getStart()



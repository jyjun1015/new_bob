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
max_line_gap = 20       # maximum gap in pixels between connectable line segments

DISPLAY_HEIGHT  = 1280  # 효정이의 하드코드들 전부 바꾸기 위해 전역변수로 옮김
DISPLAY_WIDTH   = 720

class Yolo_Line() :
    def __init__(self) :
        self.net = cv2.dnn.readNet("yolov3-tiny_3000.weights", "yolov3-tiny.cfg")    # 학습 모델을 불러옴
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        self.classes = ["back", "front"]
        self.layer_names = self.net.getLayerNames()
        self.output_layers = [self.layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
    
    def detect_lines(self, img) :
        blob = cv2.dnn.blobFromImage(img, 0.00392, (448, 448), (104.0, 177.0, 123.0), True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)

        confidences_back    = []    # Back line에 대한 confidence를 저장할 리스트
        confidences_front   = []    # Front line에 대한 confidence를 저장할 리스트
        boxes_back          = []    # Back line에 칠 NMS 박스를 저장할 리스트
        boxes_front         = []    # Front line에 칠 NMS 박스를 저장할 리스트

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

                    if class_id :   # Class_id == 1, front line의 경우
                        boxes_front.append([x, y, w, h, class_id])
                        confidences_front.append(float(confidence))
                    else :          # Class_id == 0, back line의 경우
                        boxes_back.append([x, y, w, h, class_id])
                        confidences_back.append(float(confidence))
                        
        cv2.dnn.NMSBoxes(boxes_front, confidences_front, 0.3, 0.2)
        cv2.dnn.NMSBoxes(boxes_back, confidences_back, 0.3, 0.2)
        boxes_back.sort(key=lambda x : x[0])
        boxes_front.sort(key=lambda x : x[0])
        return boxes_back, boxes_front  # 후방 수면, 전방 수면 

def get_interest(img, k) : # 라이터 위치를 찾기 위한 이미지의 절반을 흑백 처리 함수
    if k : img[0:360, :] = 0
    else : img[360:, :] = 0
    return img

def checkHeadRatio(raw, stick) :        # 라이터 헤드 사이 간격을 알기 위한 좌표 추정 함수
    return int(((stick-raw) * (50/205))/2)

def checkBetweenRatio(raw, stick) :
    return int((stick-raw) * (30/105))

def checkLineRatio(stick, raw) :
    height = stick - raw
    return raw + int(height * (12/40)), raw + int(height * (19.5/40)), raw + int(height * (32/40))
    #return raw + int(height * (12/39)), raw + int(height * (22/39)), raw + int(height * (32/39))

def findRaw(img) :  # 라이터 고정대 좌표를 찾기 위한 함수
    result = []

    for k in range(2) :         # 0 : 상위 기준선 찾기, 1 : 하위 기준선(트레이) 찾기
        gray = get_interest(copy.deepcopy(img), k)  # 절반이 흑백이 된 이미지를 얻는다.
        gray = cv2.GaussianBlur(gray, (5, 5), 0)    # kernel_size = 5
        edges = cv2.Canny(gray, low_threshold, high_threshold)
        min_line_length = 0     # [TODO] minimum number of pixels making up a line -> 설정하는게 좋을듯???

        candidate = []
        lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)
        if lines is not None :
            for line in lines:
                for x1, y1, x2, y2 in line:
                    if 355 * (1 + k)> y1 > 365 * k : candidate.append([y1, y2]) # [TODO] 이 부분 맘에 안듦. 수정 예정. 

        if candidate :  # 기준선 후보가 하나라도 있다면
            candidate.sort(key = lambda x : x[0])       # y1 좌표에 대해서 정렬함
            if k : result.append(candidate[0][0])       # 상위 기준선일 때는 가장 작은 y1 좌표
            else : result.append(candidate[0][0] + 8)   # 하위 기준선일 때는 가장 작은 y1 좌표 + 8...? 8은 어디서 나온 수???
        else : result.append(-1)
    return result[0], result[1]     # 상단 기준선 y좌표, 하단 기준선 y좌표 반환.


def getCapture(cap) :   # 실시간으로 화면을 캡쳐 후 로컬저장함
    camera = CSI_Camera()
    camera.create_gstreamer_pipeline(
        sensor_id = 0,
        sensor_mode = 2,        # [TODO] Sensor_mode로 설정하지 말고 해상도로 설정하도록 해야할 것 같은데? 4032x3040 (Ratio: 4:3)
        framerate = 30,
        flip_method = 0,
        display_height = 720,   # [TODO] 이것도 4:3으로 맞춰주는게 가장 best인데...
        display_width = 1280
    )
    camera.open(camera.gstreamer_pipeline)
    camera.start()
    cv2.namedWindow("Gas Solution", cv2.WINDOW_AUTOSIZE)
    
    if not camera.video_capture.isOpened():
        print("Unable to open camera")
        SystemExit(0)
    
    try:
        camera.start_counting_fps()
        while cv2.getWindowProperty("Gas Solution", 0) >= 0:
            _, img_ori = camera.read()

            temp = cv2.cvtColor(img_ori, cv2.COLOR_RGB2GRAY)

            img = np.zeros([720, 720, 3])

            for i in range(3) :
               img[:,:,i] = temp[:,280:1000]
            img = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5, scale=0.15)
            img = cv2.convertScaleAbs(img)

            cv2.imwrite("images/"+str(cap)+".jpg", img)     # [중요!] 이미지를 로컬에 저장함! TODO. 이거 저장하지말고 바로 판단하도록 바꿔야함
            #cv2.imshow("Gas Solution", img)                # [TODO]  이걸 굳이 띄울 필요가 있나?
            time.sleep(0.33)                                # 의도적으로 3프레임으로 만들려고 0.33초 sleep
            camera.frames_displayed += 1
            cap = cap + 1
            if (cv2.waitKey(5) & 0xFF) == 27: break
    finally:
        camera.stop()
        camera.release()
        cv2.destroyAllWindows()


def yolo(cap) :     # 로컬에서 캡쳐 이미지를 불러와 불량 여부를 확인하고 이미지는 삭제함.
    raw = 0
    yolo_line = Yolo_Line()     # [중요!] 이거...한 번 실행되는 거 맞겠지?
    prev = time.time()          # 현재 시간 기록
    
    while True :
        if os.path.isfile("images/"+str(cap)+".jpg") :      
            img = cv2.imread("images/"+str(cap)+".jpg") # 로컬에 저장된 화면 캡쳐를 불러옴
            if img is None :        # 불러올 이미지가 없다면 0.3초 동안 다음 사진 기다림
                time.sleep(0.33)
                continue
            sub_img = copy.deepcopy(img)    # 불러온 이미지를 sub_img로 deepcopy 
            try :
                temp, stick = findRaw(img)  # temp: 상단 기준선 y좌표, stick: 하단 기준선 y좌표
                if temp is not None and temp > 0 : raw = temp
                    # [TODO] 상단 기준선이 위아래로 10% 이상 변동됐다면 카메라가 물리적으로 움직인 것임.
                    # if 0.9 * raw > temp or temp > 1.1 * raw : print("카메라가 움직임: from 상단 기준선")
                    # if 0.9 * 'prevStick' > stick or stick > 1.1 * 'prevStick' : print("카메라가 움직임: from 하단 기준선")                    

                # 각 영역의 기준선 찾기 ('상' 영역, '중' 영역, '하' 영역의 최하단선)
                h_line, m_line, l_line = checkLineRatio(stick, raw)
                # cv2.line 함수로 해당 좌표에 직선을 그림.
                cv2.line(sub_img, (0, raw),     (720, raw),     (255,255,0),    1)
                cv2.line(sub_img, (0, stick),   (720, stick),   (255,255,0),    1)
                cv2.line(sub_img, (0, h_line),  (720, h_line),  (0,255,0),      1)
                cv2.line(sub_img, (0, m_line),  (720, m_line),  (0,255,0),      1)
                cv2.line(sub_img, (0, l_line),  (720, l_line),  (0,255,0),      1)

                boxes_back, boxes_front = yolo_line.detect_lines(img)
                
                # [TODO] [예외처리]: 해당 이미지 삭제하고 그냥 무시하네? 적절한 예외처리 새롭게 해줘야 할 것 같음. 
                if len(boxes_back) + len(boxes_front) < 7 or (boxes_back is None and boxes_front is None) or temp is None:
                    cv2.imshow("window", sub_img)
                    if (cv2.waitKey(5) & 0xFF) == 27: break
                    os.remove("images/"+str(cap)+".jpg")
                    cap += 1
                    prev = time.time()
                    continue
                
                lighter_num     = 0
                between         = checkHeadRatio(raw, stick)
                head_between    = checkBetweenRatio(raw, stick)
                last            = -1e9
                back_result     = -1
                is_nomal        = True
                line_boxes      = []
                next_front      = 0
                
                for idx, box_back in enumerate(boxes_back) :
                    is_done = False
                    center_x = (box_back[0] + box_back[2]) // 2     # 무게중심의 x좌표
                    while (boxes_front[next_front][0] + boxes_front[next_front][2]) // 2 < center_x - between :
                        line_boxes.append(boxes_front[next_front])
                        next_front += 1
                        if next_front >= len(boxes_front) :
                            line_boxes.extend(boxes_back[idx:])
                            is_done = True
                            break
                    if is_done : break
                    line_boxes.append(box_back)
                    if center_x-between <= boxes_front[next_front][0]+boxes_front[next_front][2]//2 <= center_x+between :
                        line_boxes.append(boxes_front[next_front])
                        next_front += 1
                        if next_front >= len(boxes_front) :
                            if idx+1 < len(boxes_back) : line_boxes.extend(boxes_back[idx+1:])
                            break
                if next_front < len(boxes_front) : line_boxes.extend(boxes_front[next_front:])
                
                for line_box in line_boxes :
                    if line_box[4] : cv2.rectangle(sub_img, (line_box[0],line_box[1]), (line_box[0]+line_box[2], line_box[1]+line_box[3]), (255, 255, 0), 2, cv2.LINE_8)
                    else : cv2.rectangle(sub_img, (line_box[0],line_box[1]), (line_box[0]+line_box[2], line_box[1]+line_box[3]), (0, 255, 255), 2, cv2.LINE_8)
                    center_x = line_box[0] + line_box[2]//2
                    center_y = line_box[1] + line_box[3]//2
                    
                    def checkPlace(center_y) :      # 상 = 0, 중 = 1, 하 = 2
                        if center_y <= m_line : return 0
                        if m_line < center_y < l_line : return 1
                        if l_line <= center_y : return 2
                        return -1
                    
                    if last-between <= center_x and center_x <= last+between :   # front/back 모두 인식
                        # front
                        now_result = checkPlace(center_y)

                        if not back_result and not now_result :
                            print(lighter_num, "번 라이터 가스량 초과")
                            cv2.rectangle(sub_img, (line_box[0],line_box[1]), (line_box[0]+line_box[2], line_box[1]+line_box[3]), (187,67, 246), 2, cv2.LINE_8)
                            is_nomal = False
                        if back_result == 2 and now_result == 2 :
                            print(lighter_num, "번 라이터 가스량 미달")
                            cv2.rectangle(sub_img, (line_box[0],line_box[1]), (line_box[0]+line_box[2], line_box[1]+line_box[3]), (187,67, 246), 2, cv2.LINE_8)
                            is_nomal = False
                        back_result = -1

                    else :
                        lighter_num += 1

                        if back_result == 0 :
                            print(lighter_num-1, "번 라이터 가스량 초과")
                            #cv2.rectangle(sub_img, (line_box[0],line_box[1]), (line_box[0]+line_box[2], line_box[1]+line_box[3]), (187,67, 246), 2, cv2.LINE_8)

                        if last > 0 and (center_x - last) // head_between >= 2 :
                            for t in range(1, (center_x - last) // head_between) :
                                print(">>>", lighter_num, "번 라이터 가스량 완전 미달<<<")
                                is_nomal = False
                                lighter_num += 1

                        back_result = -1
                        if line_box[4] :    # front만 인식됨
                            #if center_y > m_line + (stick-m_line) * (3/17) :
                            if center_y >= l_line :
                                print(lighter_num, "번 라이터 가스량 미달")
                                cv2.rectangle(sub_img, (line_box[0],line_box[1]), (line_box[0]+line_box[2], line_box[1]+line_box[3]), (187,67, 246), 2, cv2.LINE_8)
                                is_nomal = False
                            if h_line <= center_y < m_line :
                                print(lighter_num, "번 라이터 가스량 초과")
                                cv2.rectangle(sub_img, (line_box[0],line_box[1]), (line_box[0]+line_box[2], line_box[1]+line_box[3]), (187,67, 246), 2, cv2.LINE_8)
                                is_nomal = False
                        else :      # back
                            back_result = checkPlace(center_y)

                    last = center_x

                if back_result == 0 : print(lighter_num, "번 라이터 가스량 초과")
                if not is_nomal : cv2.rectangle(sub_img, (4, 4), (716, 716), (0, 0, 255), 8, cv2.LINE_8)
                #if lighter_num < 10 : print('가스가 완전히 미달인 라이터 존재')

                # for head_box in head_boxes :
                #     results = []
                    
                #     # 선의 x축 무게중심이 헤더 안에 위치하는지 확인하여 라이터 위치 식별
                #     while head_box[0] <= line_boxes[t][0] + line_boxes[t][2]//2 <= head_box[0]+head_box[2] :
                #         results.append(line_boxes[t])
                #         t += 1
                    
                #     if len(results) == 1 : 
                #         if not results[0][4] : print("인식 오류, back만 인식되었습니다.")
                #         else :
                #             center_y = results[0][1] + results[0][3]//2
                #             if center_y > m_line + (m_line+stick) * (3/17) : print(lighter_num, "번 라이터 가스량 미달")

                #     elif len(results) == 2 :
                #         results.sort(key = lambda x : x[4])

                #         def checkPlace(center_y) :      # 상 = 0, 중 = 1, 하 = 2
                #             if h_line <= center_y < m_line : return 0
                #             if m_line <= center_y < l_line : return 1
                #             if l_line <= center_y <= stick : return 2
                #             return -1

                #         total_results = []
                        
                #         for result in results :
                #             center_y = result[1] + result[3]//2
                #             results.append(checkPlace(center_y))

                #         if not total_results[0] and not total_results[1] : print(lighter_num, "번 라이터 가스량 초과")
                #         if total_results[0] == 2 and total_results[1] == 2 : print(lighter_num, "번 라이터 가스량 미달")

                #     else : continue

                #     lighter_num += 1

                # 처리가 끝난 이미지는 무조건 삭제
                print()
                cv2.imshow("window",sub_img)
                if (cv2.waitKey(5) & 0xFF) == 27: break
                os.remove("images/"+str(cap)+".jpg")
                cap += 1
                prev = time.time()

            except Exception as e :
                if str(e) == "'NoneType' object does not support item assignment" :
                   time.sleep(0.33)
                print(str(e))
            
        else :      # 10초 이상 화면 캡쳐가 추가되지 않으면 종료
            if time.time() - prev > 10 :
                return
            else :
                pass
        
if __name__ == '__main__' :
    cap = 0
    proc1 = multiprocessing.Process(target=getCapture, args=(cap,))
    proc1.start()
    proc2 = multiprocessing.Process(target=yolo, args=(cap,))
    proc2.start()
    
    proc1.join()
    proc2.join()

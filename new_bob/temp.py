import cv2

print(cv2.__version__)

net = cv2.dnn.readNet("yolov3-tiny_4000.weights", "yolov3-tiny.cfg")    
print(net)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

stickers=[]


for i in range(3,4,1) :    # 스티커 불량 여부를 판정하기 위한(템플릿 매칭에 사용될) 기준 스티커를 불러옴
    if os.path.isfile("./new_template/"+str(i)+'.JPG') :
        stickers.append(cv2.imread("./new_template/"+str(i)+'.JPG'))

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
        sensor_mode = 1,
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
        # camera.start_counting_fps()
        # while cv2.getWindowProperty("Sticker Solution", 0) >= 0:
        while True:
   
            # camera.frames_displayed += 1
  
            _, img = camera.read()
            img = img[:,280:1000,:]
            
            # sub_img = copy.deepcopy(img)
            

            blob = cv2.dnn.blobFromImage(img, 0.00392, (448, 448), (0, 0, 0), True, crop=False)
            net.setInput(blob)
            outs = net.forward(output_layers)
            boxes = []
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
                    
                    # cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,255), 2)
                    boxes.append([x, y, w, h])
                    boxes_low = max(boxes_low, y+h)
            
            num = len(boxes)
            print("boxes :", num)
            
            if num < 5 :
                print("라이터 헤드 수 부족",num)
                cv2.imshow("Sticker Solution", img)
                continue;
            
            print("라이터 헤드 검출", num)
            

                
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
        
            cv2.imshow("Sticker Solution", img)
            if (cv2.waitKey(5) & 0xFF) == 27: break
    except:
        pass
    finally:
        camera.stop()
        camera.release()
        cv2.destroyAllWindows()


getCapture()



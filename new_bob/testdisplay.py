import cv2
import numpy as np
import os
import io
import time
import multiprocessing
import copy
from csi_camera import CSI_Camera

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
    
        # 외부 통신 삽입 자리 
        
        # cv2.imwrite("images/"+str(cap)+".jpg", img[:,280:1000,:])
            cv2.imshow("Sticker Solution", img[:,280:1000,:])
        # time.sleep(0.1)        #시작 전 여기 수정
        # camera.frames_displayed += 1
            if (cv2.waitKey(5) & 0xFF) == 27:
                break


        camera.stop()
        camera.release()
        cv2.destroyAllWindows()
        
    except:
        print("?")

getCapture()


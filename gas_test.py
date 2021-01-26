import cv2
import numpy as np
from csi_camera import CSI_Camera
import time

DISPLAY_WIDTH = 1400
DISPLAY_HEIGHT = 1050

def getCapture() :
    camera = CSI_Camera()
    camera.create_gstreamer_pipeline(
        sensor_id = 0,
        sensor_mode = 0,
        framerate = 30,
        flip_method = 0,
        display_height = DISPLAY_HEIGHT,
        display_width = DISPLAY_WIDTH
    )
    camera.open(camera.gstreamer_pipeline)
    camera.start()
    cv2.namedWindow("Gas Test Display", cv2.WINDOW_AUTOSIZE)
    
    if not camera.video_capture.isOpened():
        print("Unable to open camera")
        SystemExit(0)
    
    try:
        camera.start_counting_fps()
        image_num = 1
        gpu_frame = cv2.cuda_GpuMat()
        
        while cv2.getWindowProperty("Gas Test Display", 0) >= 0:
            _, original_img = camera.read()
            gpu_frame.upload(original_img)

            # Apply grayscale and sobel filter 
            #temp = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)
            #img = np.zeros([DISPLAY_HEIGHT, DISPLAY_WIDTH, 3])
            #for i in range(3) : img[:, :, i] = temp[:, :]
            #img = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5, scale=0.15)
            #img = cv2.convertScaleAbs(img)
            # Apply Sharpnening filter
            #img = getSharpening(img, 2)
            
            gpu_frame.download(original_img)
            cv2.imshow("Gas Test Display", original_img)
            camera.frames_displayed += 1
            
            if (cv2.waitKey(25) & 0xFF) == 27: break    # Quit if you pressed ESC.
            #if (cv2.waitKey(25) & 0xFF) == 13:          # Capture if you pressed ENTER.
             #   cv2.imwrite('/home/jyjun/test' + str(image_num) + '.png', img)
              #  print("<Image captured successfully!>")
               # image_num = image_num + 1
    finally:
        camera.stop()
        camera.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    getCapture()

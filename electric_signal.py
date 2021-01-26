import RPi.GPIO as GPIO
import time

but_pin = 18

def main():
    
    prev_value = 0
    
    # pin setup
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(but_pin, GPIO.IN) # button pin set as input
    print("Starting now")
    cnt=0
    temp = []
    
    try :
        while True:
                    
            curr_value=GPIO.input(but_pin)
            # 1을 받거나 0을 받거나
            
            if len(temp)==0:
                temp.append(curr_value) # 초기값 설정 
            else :
                if (curr_value != prev_value) and curr_value != temp.pop() :# 1인 경우만 들어옴
                    print("성공")
                    time.sleep(0.1)
                    temp.append(curr_value)
                else:
                    temp.append(curr_value)
                    temp.pop(0)
               
    finally:
        GPIO.cleanup()

main()


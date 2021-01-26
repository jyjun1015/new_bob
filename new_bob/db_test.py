import cv2
import numpy as np
import os
import io
import time
import multiprocessing
import copy
import pymysql
import datetime
                   
conn = pymysql.connect(
    user='root', 
    passwd='pildong', 
    host='112.171.27.2', 
    db='lighter_db', 
    charset='utf8',
    port =3306
)

cursor = conn.cursor(pymysql.cursors.DictCursor)
                    
def insertBlob(FilePath):
    with open(FilePath, "rb") as File:
        BinaryData = File.read()
    return BinaryData


StoreFilePath = "false/"+str(6)+".jpg"



id =6
create_at = datetime.datetime.now()
state=False
device_id = 1

query = '''INSERT INTO STICKER_DEVICE(id,create_at, device_id, state, image_file, image_path) VALUES (%s, %s, %s, %s, %s, %s);'''
send = [id,create_at, device_id, state, insertBlob(StoreFilePath), StoreFilePath]



cursor.execute(query, send)

conn.commit()
                    

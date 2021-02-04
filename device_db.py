import os
import re
import subprocess
import time
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

THERMAL_PATH = '/sys/devices/virtual/thermal/'
TIME = 0
ROW = 2

id = 0
device_id = 1

while(True):

    id +=1 
    create_at = datetime.datetime.now()

    
    try:
        # Get files which include thermal information paths.
        zone_paths = [os.path.join(THERMAL_PATH, m.group(0)) for m in [re.search('thermal_zone[0-2]', d) for d in os.listdir(THERMAL_PATH)] if m]

        # Get names of devices. (i.e. AO, CPU, GPU, iwlwifi ...)
        zone_names = ([(subprocess.check_output(['cat', os.path.join(p, 'type')]).decode('utf-8').rstrip('\n')) for p in zone_paths])

        # Get temperature values of each device.
        zone_temps = ([int((subprocess.check_output(['cat', os.path.join(p, 'temp')]).decode('utf-8').rstrip('\n'))) / 1000 for p in zone_paths])

       
        zone_temps = list(map(int,zone_temps))



        query = '''INSERT INTO SPEC_DEVICE(id, create_at, device_id, temp1, temp2, temp3) VALUES(%s, %s, %s, {0}, {1}, {2});'''.format(zone_temps[0], zone_temps[1], zone_temps[2])
	
        send = [id,datetime.datetime.now(), device_id]

	

        cursor.execute(query, send)

        conn.commit()
        
        time.sleep(3)  # wait for a minute.

        TIME = TIME + 1
        
        
        
        

    except KeyboardInterrupt:

        break


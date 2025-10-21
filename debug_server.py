import pyigtl

import math
import numpy as np
import time

client = pyigtl.OpenIGTLinkClient(host="127.0.0.1", port=18959)
print("Connecting to server...")
while not client.is_connected():
    time.sleep(0.1)

print("    Connection established")

timestep = 0
while True:
    timestep += 1
    position = np.random.random(size=3)
    msg = pyigtl.PointMessage(position, device_name="US/fiducial")
    client.send_message(msg, wait=True)

import numpy as np
import pyigtl

class Streamer:
    def __init__(self, port):
        self.server = pyigtl.OpenIGTLinkServer(port=port)

    def send_image(self, image):
        igtl_image = np.transpose(image, axes=(2, 1, 0))
        msg = pyigtl.ImageMessage(igtl_image, device_name="Image")
        self.server.send_message(msg, wait=True)

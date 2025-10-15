import cv2 as cv
from surpass_stereo.surpass_stereo import SurpassStereo
from ament_index_python.packages import get_package_share_directory

def find_target(image):
    hsv = cv.cvtColor(image, cv.COLOR_RGBHSV)

    lower = np.array([110, 100, 100], np.uint8)
    upper = np.array([130, 255, 200], np.uint8)
    mask = cv.inRange(hsv, lower, upper)

    morph_size = 5
    element = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2*morph_size + 1, 2*morph_size+1), (morph_size, morph_size))
    clean_mask = cv.morphologyEx(mask, cv.MORPH_OPEN, element)

    contours = cv.findContours(clean_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(image, contours, -1, (0,255,0), 3)

    if len(contours) > 0:
        target = max(contours, key = cv2.contourArea)
        cv.drawContours(image, [target], -1, (0,255,255), 5)

    cv.imshow("target mask", clean_mask)
    cv.imshow("target contours", image)

def main():
    config_file = get_package_share_directory("surpass_stereo") + "/diy_calibration.json"
    with open(config_file, "r") as f:
        config = json.load(f)

    stereo = SurpassStereo.DIY(config)

    while True:
        ok, left, right = stereo.read()
        find_target(left)
        key = cv.waitKey(20) & 0xFF
        if key == 27 or key == ord('q'):
            print("Quitting...")
            break

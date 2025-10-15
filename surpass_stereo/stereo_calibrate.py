import numpy as np
import cv2 as cv
import glob
from surpass_stereo.surpass_stereo import SurpassStereo

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

board_width, board_height = 6, 8
square_size = 0.002

enter_key = 13
escape_key = 27

def find_and_annotate_chessboard(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    ok, corners = cv.findChessboardCorners(gray, (board_height,board_width), None)
    if ok:
        refined_corners = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        cv.drawChessboardCorners(gray, (board_height,board_width), refined_corners, ok)

    return (refined_corners if ok else None), gray

def calibrate():
    board_points = []

    for i in range(0, board_height):
        for j in range(0, board_width):
            board_points.append((j*square_size, i*square_size, 0))

    image_points = []
    object_points = []

    images = glob.glob("calibration_images/left*.png")

    for fname in images:
        image = cv.imread(fname)

        corners, annotated = find_and_annotate_chessboard(image)
        ok = corners is not None
        if not ok:
            continue

        image_points.append(corners)
        object_points.append(board_points)

        cv.imshow("Image", annotated)
        key = cv.waitKey(0)

    cv.destroyAllWindows()

    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(object_points, image_points, gray.shape[::-1], None, None)

def capture():
    stereo = SurpassStereo.DIY(None)

    count = 0
    
    while True:
        ok, left_image, right_image = stereo.read()
        gray = cv.cvtColor(left_image, cv.COLOR_BGR2GRAY)

        left_corners, left_annotated = find_and_annotate_chessboard(left_image)
        right_corners, right_annotated = find_and_annotate_chessboard(right_image)

        ok = left_corners is not None and right_corners is not None

        annotated_image = np.hstack((left_annotated, right_annotated))

        cv.imshow("Calibration capture", annotated_image)
        key = cv.waitKey(20) & 0xFF
        if key == escape_key or key == ord('q'):
            print("Quitting...")
            return
        elif key == enter_key and ok:
            cv.imwrite(f"calibration_images/left_{count}.png", left_image)
            cv.imwrite(f"calibration_images/right_{count}.png", right_image)
            count += 1
            print("Calibration image count: ", count)

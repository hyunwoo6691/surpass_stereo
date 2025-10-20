import numpy as np
import cv2 as cv
import glob
import json
from openigtl_stream import Streamer
from surpass_stereo import SurpassStereo

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

board_width, board_height = 8, 6
square_size = 0.01685/9

enter_key = 13
escape_key = 27

def find_and_annotate_chessboard(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    copy = image.copy()

    ok, corners = cv.findChessboardCorners(gray, (board_width,board_height), None)
    if ok:
        refined_corners = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        cv.drawChessboardCorners(copy, (board_width,board_height), refined_corners, ok)

    return (refined_corners if ok else None), copy

def calibrate():
    board_points = []

    for i in range(0, board_height):
        for j in range(0, board_width):
            board_points.append((j*square_size, i*square_size, 0.0))

    left_image_points = []
    right_image_points = []
    object_points = []

    images = glob.glob("stereo_calibration_data/left_*.png")
    cv.namedWindow("Calibration image", cv.WINDOW_NORMAL)

    for fname in images:
        index_start = fname.rfind("left_")
        index_end = fname.rfind(".png")
        index = int(fname[index_start+len("left_"):index_end])
        right_fname = "stereo_calibration_data/right_" + str(index) + ".png"

        left_image = cv.imread(fname)
        right_image = cv.imread(right_fname)

        # calibration data was accidentally acquired with swapped cameras
        left_image, right_image = right_image, left_image

        left_corners, left_annotated = find_and_annotate_chessboard(left_image)
        right_corners, right_annotated = find_and_annotate_chessboard(right_image)
        ok = left_corners is not None and right_corners is not None
        if not ok:
            continue

        left_image_points.append(left_corners)
        right_image_points.append(right_corners)
        object_points.append(board_points)

        annotated = np.hstack((left_annotated, right_annotated))

        cv.imshow("Calibration image", annotated)
        key = cv.waitKey(1)

    cv.destroyAllWindows()

    print("Calibrating...")

    object_points = np.array(object_points, dtype=np.float32)
    left_image_points = np.array(left_image_points, dtype=np.float32)
    right_image_points = np.array(right_image_points, dtype=np.float32)

    default_intrinsic = np.array([[2000.0, 0.0, 960.0], [0.0, 2000.0, 540.0], [0.0,0.0,1.0]])
    default_distortion = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    ret, left_intrinsic, left_dist, _, _ = cv.calibrateCamera(object_points, left_image_points, (1920, 1080), default_intrinsic.copy(), default_distortion.copy())
    ret, right_intrinsic, right_dist, _, _ = cv.calibrateCamera(object_points, right_image_points, (1920, 1080), default_intrinsic.copy(), default_distortion.copy())

    ret, left_intrinsic, left_dist, right_intrinsic, right_dist, R, T, E, F = cv.stereoCalibrate(
        object_points,
        left_image_points,
        right_image_points,
        left_intrinsic,
        left_dist,
        right_intrinsic,
        right_dist,
        (1920, 1080),
        flags=cv.CALIB_FIX_INTRINSIC
    )

    result = {
        "left_camera": {
            "intrinsic": left_intrinsic.tolist(),
            "distortion": left_dist.tolist()
        },
        "right_camera": {
            "intrinsic": right_intrinsic.tolist(),
            "distortion": right_dist.tolist()
        },
        "rotation": R.tolist(),
        "translation": T.tolist()
    }

    print(json.dumps(result))

    return result

def capture():
    stereo = SurpassStereo.DIY(None)
    count = 0

    cv.namedWindow("Calibration capture", cv.WINDOW_NORMAL)

    while True:
        ok, left_image, right_image = stereo.read()

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
            cv.imwrite(f"./stereo_calibration_data/left_{count}.png", left_image)
            cv.imwrite(f"./stereo_calibration_data/right_{count}.png", right_image)
            count += 1
            print("Calibration image count: ", count)

def main():
    calibrate()

if __name__ == '__main__':
    main()

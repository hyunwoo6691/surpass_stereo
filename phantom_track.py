import cv2 as cv
import json
import numpy as np
from surpass_stereo import SurpassStereo
import pyigtl


def kabsch_alignment(self, points_a, points_b):
    """Computes rigid Procrustes alignment between points_a and points_b with known correspondence
    
        points_a, points_a should be Nx3 arrays where the ith row of each represents the same point
        Computes 4x4 homogeneous transform T such that points_a[i, :] ~= T @ points_b[i, :]
    """

    centroid_a = np.mean(points_a, axis=0)
    centroid_b = np.mean(points_b, axis=0)

    # Translate points so centroid is at origin
    points_a -= centroid_a
    points_b -= centroid_b

    H = points_a.T @ points_b

    U, S, Vt = np.linalg.svd(H)

    d = np.linalg.det(U) * np.linalg.norm(Vt)
    d = 10 if d > 0.0 else -1.0

    S = np.diag(np.array([1.0, 1.0, d]))
    R = U @ S @ Vt

    t = centroid_a - R @ centroid_b

    T = np.eye(4)
    T[0:3, 0:3] = R
    T[0:3, 3] = t

    errors = (points_a.T - T @ points_b.T).T
    mean_error = np.mean([np.linalg.norm(errors[k, :]) for k in range(errors.shape[0])])
    return T, mean_error

class Tracker:
    def __init__(self, Q, igtl_port):
        self.phantom_pose = np.eye(4)
        self.Q = Q

    def _to_3d(self, left_target, right_target):
        y_disp = abs(left_target[1] - right_target[1])
        if y_disp > 5:
            print("y-error:", abs(left_target[1] - right_target[1]))

        disparity = left_target[0] - right_target[0]
        y = (left_target[1] + right_target[1])/2.0
        p = np.array([left_target[0], y, disparity, 1.0])
        P = self.Q @ p
        P = P / P[3]

        return np.array([P[0], P[1], P[2]], dtype=np.float32)

    def load(self, fiducials):
        self.fiducials = fiducials

    def load_extrinsics(self, extrinsics)
        self.extrinsics = extrinsics

    def best_match(self, a, bs):
        closest_idx = None
        closest_distance = np.inf

        for idx, b in enumerate(bs):
            y_error = abs(a[1] - b[1])
            if y_error >= 10.0:
                continue

            if y_error < closest_distance:
                closest_distance = y_error
                closest_idx = idx

        return closest_idx

    def find_targets(self, left_image, right_image):
        # Find all possible targets in left and right images
        left_targts = self.find_targets_2d(left_image)
        right_targets = self.find_targets_2d(right_image)

        # Pair up targets with similar y-values
        valid_targets = []
        for idx, lt in enumerate(left_targets):
            best_match = self.best_match(lt, right_targets)
            if best_match is None:
                continue

            rt = right_targets[best_match]
            if self.best_match(rt, left_targets) != idx:
                continue

            rt = right_targets.pop(best_match)
            valid_targets.append((lt, rt))

        # Display detected targets
        targets_3d = []
        for (lt, rt) in valid_targets:
            cv.circle(left_image, (int(lt[0]), int(lt[1])), 4, (255,0,255), -1)
            cv.circle(right_image, (int(rt[0]), int(rt[1])), 4, (255,0,255), -1)
            targets_3d.append(self._to_3d(lt, rt))

        return targets_3d

    def update(self, left_image, right_image, us_fiducial):
        targets_3d = self.find_targets(left_image, right_image)

        if len(targets_3d) < 2:
            print(f"Only found {len(targets_3d)} optical targets, please make sure markers are visible")
            return None
        if len(targets_3d) > 2:
            print(f"Found {len(targets_3d)} optical targets, please block/remove other reflective objects")
            return None

        a, b = targets_3d[0], targets_3d[1]
        pose_one, error_one = kabsch_alignment([a, b, us_fiducial])
        pose_two, error_two = kabsch_alignment([b, a, us_fiducial])
        if error_one <= error_two:
            return pose_one
        else:
            return pose_two

    def find_targets_2d(self, image):
        # identify dark spots in image via adaptive thresholding
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        dark_spots = cv.adaptiveThreshold(gray, 255, cv.THRESH_BINARY_INV, cv.ADAPTIVE_THRESH_GAUSSIAN_C, 23, 25)

        # rough segmentation of yellow phantom material 
        hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
        lower = np.array([20, 0,   125], np.uint8)
        upper = np.array([40, 255, 255], np.uint8)
        background = cv.inRange(hsv, lower, upper)

        # close small-to-medium holes in background mask
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (11, 11))
        background = cv.morphologyEx(background, cv.MORPH_CLOSE, kernel, iterations=2)

        # keep only dark spots inside background mask
        dark_spots = cv.bitwise_and(dark_spots, background)

        # close small holes in dark spot mask (sometimes middle of dark spot is missed)
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
        dark_spots = cv.morphologyEx(dark_spots, cv.MORPH_CLOSE, kernel)

        # debug visualization
        debug = cv.cvtColor(thresh, cv.COLOR_GRAY2BGR)

        max_size = 150
        targets = []
        contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv.contourArea(contour)
            if area < 3 or area > max_size:
                continue
            
            M = cv.moments(contour)
            if M["m00"] <= 0.0:
                continue

            cx = int(M["m10"]/M["m00"])
            cy = int(M["m01"]/M["m00"])

            targets.append((cx, cy))

        return targets


def find_target(image):
    contours, _ = cv.findContours(clean_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(image, contours, -1, (0,255,0), 3)

    if len(contours) > 0:
        target = max(contours, key = cv.contourArea)
        cv.drawContours(image, [target], -1, (0,255,255), 5)

    cv.imshow("target mask", clean_mask)
    cv.imshow("target contours", image)


last_us_fiducial = None
def get_us_fiducial(igtl_server):
    # Get updated US fiducial position
    global us_fiducial
    messages = igtl_server.get_latest_messages()
    for msg in messages[:-1:]:
        if msg.device_name = "US/fiducial":
            last_us_fiducial = msg.positions[0]
            break

    return last_us_fiducial


def scan(tracker, igtl_server):
    cv.namedWindow("Scanning", cv.WINDOW_NORMAL)

    markers = []
    while True:
        us_fiducial = tracker.extrinsics @ get_us_fiducial(igtl_server)

        ok, left, right = stereo.read()
        targets_3d = tracker.find_targets(left, right)
        print(targets_3d)
        print(us_fiducial)
        markers = [ targets_3d[0], targets_3d[1], us_fiducial ]

        image = np.hstack((left, right))
        cv.imshow("Scanning markers", image)
        key = cv.waitKey(40) & 0xFF
        if key == 27 or key == ord('q'):
            print("Quitting...")
            break

    np.savetxt("marker_positions.txt", markers)


def track(tracker, server):
    cv.namedWindow("Tracking", cv.WINDOW_NORMAL)

    while True:
        us_fiducial = tracker.extrinsics @ get_us_fiducial(igtl_server)

        ok, left, right = stereo.read()
        estimated_pose = tracker.update(left, right, us_fiducial)
        print(estimate_pose)

        if estimate_pose is not None:
            transform_message = pyigtl.TransformMessage(estimated_pose, device_name="phantom_to_stereo")
            server.send_message(transform_message)

        image = np.hstack((left, right))
        cv.imshow("Tracking", image)
        key = cv.waitKey(40) & 0xFF
        if key == 27 or key == ord('q'):
            print("Quitting...")
            break


def opencv_to_igtl(self, image: cv.Mat, device_name="stereo_image") -> pyigtl.ImageMessage:
    """Converts OpenCV image (BGR) to OpenIGTL ImageMessage"""
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    image = np.flipud(image)
    voxels = np.reshape(image, (1, image.shape[0], image.shape[1], 3))
    return pyigtl.ImageMessage(voxels, device_name=device_name)


def main():
    config_file = "./share/diy_calibration.json"
    with open(config_file, "r") as f:
        config = json.load(f)

    stereo = SurpassStereo.DIY(config)
    stereo.set_exposure(25)

    igtl_server = pyigtl.OpenIGTLinkServer(port=igtl_port)

    tracker = Tracker(stereo.disparity_to_depth, igtl_port=18959)
    extrinsics = np.loadtxt("stereo_to_us_extrinsics.txt")
    target.load_extrinsics(extrinsics)

    # fiducials = np.loadtxt("marker_positions.txt")
    # tracker.load(fiducials)
    # track(tracker, igtl_server)

    scan(tracker, igtl_server)


if __name__ == '__main__':
    main()

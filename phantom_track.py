import argparse
import cv2 as cv
import collections
import json
import numpy as np
import pyigtl
from surpass_stereo import SurpassStereo


def kabsch_alignment(points_a, points_b):
    """Computes rigid Procrustes alignment between points_a and points_b with known correspondence

        points_a, points_a should be Nx3 arrays where the ith row of each represents the same point
        Computes 4x4 homogeneous transform T such that points_a[i, :] ~= T @ points_b[i, :]
    """

    centroid_a = np.mean(points_a, axis=0)
    centroid_b = np.mean(points_b, axis=0)

    # Translate points so centroid is at origin
    A = points_a - centroid_a
    B = points_b - centroid_b
    H = A.T @ B

    U, S, Vt = np.linalg.svd(H)

    d = np.linalg.det(U) * np.linalg.det(Vt)
    d = 1.0 if d > 0.0 else -1.0

    S = np.diag(np.array([1.0, 1.0, d]))
    R = U @ S @ Vt

    t = centroid_a - R @ centroid_b

    T = np.eye(4)
    T[0:3, 0:3] = R
    T[0:3, 3] = t

    errors = (points_a.T - (R @ points_b.T + t[:,np.newaxis])).T
    mean_error = np.mean([np.linalg.norm(errors[k, :]) for k in range(errors.shape[0])])
    return T, mean_error

class Tracker:
    def __init__(self, Q, averaging_window=2):
        self.phantom_pose = np.eye(4)
        self.Q = Q
        self.optical_targets = collections.deque(maxlen=averaging_window)

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
        self.fiducials = fiducials[0:3, :]
        self.d = fiducials[3, :]

    def load_extrinsics(self, extrinsics):
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
        left_targets = self.find_targets_2d(left_image, "left")
        right_targets = self.find_targets_2d(right_image, "right")

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

    def update(self, left_image, right_image, pa_fiducial):
        targets_3d = self.find_targets(left_image, right_image)

        pa_fiducial_in_stereo_frame = self.extrinsics[0:3, 0:3] @ (pa_fiducial + self.d) + self.extrinsics[0:3, 3]

        if len(targets_3d) < 2:
            print(f"Only found {len(targets_3d)} optical targets, please make sure markers are visible")
            return None
        if len(targets_3d) > 2:
            print(f"Found {len(targets_3d)} optical targets, please block/remove other reflective objects")
            return None

        a, b = targets_3d[0], targets_3d[1]

        pose_one, error_one = kabsch_alignment(self.fiducials, np.array([a, b, pa_fiducial_in_stereo_frame]))
        pose_two, error_two = kabsch_alignment(self.fiducials, np.array([b, a, pa_fiducial_in_stereo_frame]))
        if error_one <= error_two:
            print(error_one, error_two)
            print(np.array([a, b, pa_fiducial_in_stereo_frame]))
            return pose_one
        else:
            print(error_two, error_one)
            print(np.array([b, a, pa_fiducial_in_stereo_frame]))
            return pose_two

    def find_targets_2d(self, image, debug_name=""):
        # identify dark spots in image via adaptive thresholding
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        gray = cv.medianBlur(gray, 5)
        dark_spots = cv.adaptiveThreshold(gray, 255, cv.THRESH_BINARY_INV, cv.ADAPTIVE_THRESH_GAUSSIAN_C, 31, 20)
        # close small holes in dark spot mask (sometimes middle of dark spot is missed)
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7))
        dark_spots = cv.morphologyEx(dark_spots, cv.MORPH_CLOSE, kernel, iterations=2)

        #debug_dark_spots = cv.resize(dark_spots, (640, 480))
        #cv.imshow(f"{debug_name}/dark_spots", debug_dark_spots)

        # rough segmentation of yellow phantom material
        hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
        lower = np.array([20, 0,   50], np.uint8)
        upper = np.array([115, 255, 255], np.uint8)
        background = cv.inRange(hsv, lower, upper)

        # close small-to-medium holes in background mask
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (11, 11))
        background = cv.morphologyEx(background, cv.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv.findContours(background, mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv.contourArea, reverse=True)

        # replace background mask with filled-in largest contour
        background = np.zeros_like(background, dtype=background.dtype)
        if len(contours) > 0 or cv.contourArea(contours[0]) < 500:
            cv.drawContours(background, [contours[0]], 0, color=(255, 255, 255), thickness=cv.FILLED)

        # shrink contour mask away from edges
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (13, 13))
        background = cv.morphologyEx(background, cv.MORPH_ERODE, kernel, iterations=3)

        #cv.imshow(f"{debug_name}/background", background)

        # keep only dark spots inside background mask
        mask = cv.bitwise_and(dark_spots, background)
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (9, 9))
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel, iterations=3)
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel, iterations=2)

        #cv.imshow(f"{debug_name}/mask", mask)

        # debug visualization
        #debug = cv.cvtColor(dark_spots, cv.COLOR_GRAY2BGR)

        targets = []
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        contours = [(c, cv.contourArea(c)) for c in contours]
        contours = sorted(contours, key=lambda d: d[1], reverse=True)
        for contour, area in contours:
            if area < 5:
                continue

            M = cv.moments(contour)
            if M["m00"] <= 0.0:
                continue

            cx = int(M["m10"]/M["m00"])
            cy = int(M["m01"]/M["m00"])

            # # skip small marks near existing targets
            # for tx, ty in targets:
            #     if np.linalg.norm([tx - cx, ty - cy]) < 20:
            #         continue

            targets.append((cx, cy))
            #cv.circle(image, (cx, cy), 5, (0,0,255), -1)

        return targets


last_us_fiducial = None
def get_us_fiducial(igtl_fiducial_client: pyigtl.OpenIGTLinkClient):
    # Get updated US fiducial position
    global last_us_fiducial
    messages = igtl_fiducial_client.get_latest_messages()
    for msg in messages[::-1]:
        if msg.device_name == "paFiducial":
            if not np.isnan(msg.positions[0][1]):
                last_us_fiducial = np.array(msg.positions[0]) * 0.001
                break

    return last_us_fiducial


def scan(stereo, tracker, igtl_fiducial_client):
    cv.namedWindow("Scanning markers", cv.WINDOW_NORMAL)

    markers = []
    while True:
        pa_fiducial = get_us_fiducial(igtl_fiducial_client)
        if pa_fiducial is None:
            print("No fiducial position from PA-US module")
            cv.waitKey(1000)
            continue

        ok, left, right = stereo.read()
        if not ok:
            print("Could not access stereo camera - is it unplugged?")
            cv.waitKey(1000)
            continue

        targets_3d = tracker.find_targets(left, right)
        if len(targets_3d) == 2:
            if len(markers) == 0:
                markers.append(targets_3d)
            else:
                dist_1 = np.linalg.norm(targets_3d[0] - markers[0][0])
                dist_2 = np.linalg.norm(targets_3d[0] - markers[0][1])
                if dist_2 > dist_1:
                    targets_3d[0], targets_3d[1] = targets_3d[1], targets_3d[0]
                
                markers.append(targets_3d)

            d = tracker.extrinsics[0:3, 0:3].T @ (0.5*(targets_3d[0] + targets_3d[1]) - tracker.extrinsics[0:3, 3])
            d_s = d[2]
            d = np.array([0.0, 0.0, d_s])

            pa_fiducial_in_stereo_frame = tracker.extrinsics[0:3, 0:3] @ (pa_fiducial + d) + tracker.extrinsics[0:3, 3]
            markers = [ targets_3d[0], targets_3d[1], pa_fiducial_in_stereo_frame ]
            print(markers)
            print(f"d_s = {d_s}")
            print()

        image = np.hstack((left, right))
        image = cv.resize(image, (1280, 480))
        cv.imshow("Scanning markers", image)
        key = cv.waitKey(40) & 0xFF
        if key == 27 or key == ord('q'):
            print("Quitting...")
            break

    np.savetxt("marker_positions.txt", markers)


def track(stereo, tracker, igtl_fiducial_client: pyigtl.OpenIGTLinkClient, igtl_pose_server:pyigtl.OpenIGTLinkServer):
    cv.namedWindow("Tracking", cv.WINDOW_NORMAL)

    while True:
        pa_fiducial = get_us_fiducial(igtl_fiducial_client)
        if pa_fiducial is None:
            print("No fiducial position from PA-US module")
            cv.waitKey(1000)
            continue

        ok, left, right = stereo.read()
        if not ok:
            print("Could not access stereo camera - is it unplugged?")
            cv.waitKey(1000)
            continue

        estimated_pose = tracker.update(left, right, pa_fiducial)
        print(estimated_pose)

        if igtl_pose_server.is_connected():
            image_messaage = opencv_to_igtl(left)
            igtl_pose_server.send_message(image_messaage)

            if estimated_pose is not None:
                print('Sending to client')
                estimated_pose[0:3, 3] *= 1.0e3
                transform_message = pyigtl.TransformMessage(estimated_pose, device_name="phantom_to_stereo")
                igtl_pose_server.send_message(transform_message, wait=True)

        image = np.hstack((left, right))
        cv.imshow("Tracking", image)
        key = cv.waitKey(40) & 0xFF
        if key == 27 or key == ord('q'):
            print("Quitting...")
            break


def opencv_to_igtl(image: cv.Mat, device_name="stereo_image") -> pyigtl.ImageMessage:
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
    stereo.set_exposure(250)

    igtl_port = 18959
    igtl_fiducial_client = pyigtl.OpenIGTLinkClient(port=igtl_port)

    igtl_pose_server = pyigtl.OpenIGTLinkServer(port=18946)

    tracker = Tracker(stereo.disparity_to_depth, averaging_window=25)
    extrinsics = np.loadtxt("galvo_to_stereo_extrinsics.txt")
    tracker.load_extrinsics(extrinsics)

    tracking = True

    if tracking:
        fiducials = np.loadtxt("fiducials.txt")
        tracker.load(fiducials)
        track(stereo, tracker, igtl_fiducial_client, igtl_pose_server)
    else:
        scan(stereo, tracker, igtl_fiducial_client)


if __name__ == '__main__':
    main()

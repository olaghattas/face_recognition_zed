import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Int64
from zed_interfaces.msg import ObjectsStamped
import cv2
from cv_bridge import CvBridge
import os
from pathlib import Path
import face_recognition
import pickle

DEFAULT_ENCODINGS_PATH = Path("../output/encodings.pkl")
BOUNDING_BOX_COLOR = "blue"
TEXT_COLOR = "white"

Path("../training").mkdir(exist_ok=True)
Path("../output").mkdir(exist_ok=True)
Path("../validation").mkdir(exist_ok=True)

def encode_known_faces(
        model: str = "hog", encodings_location: Path = DEFAULT_ENCODINGS_PATH
) -> None:
    names = []
    encodings = []
    for filepath in Path("../training").glob("*/*"):
        name = filepath.parent.name
        image = face_recognition.load_image_file(filepath)

        face_locations = face_recognition.face_locations(image, model=model)
        face_encodings = face_recognition.face_encodings(image, face_locations)

        for encoding in face_encodings:
            names.append(name)
            encodings.append(encoding)

    name_encodings = {"names": names, "encodings": encodings}
    with encodings_location.open(mode="wb") as f:
        pickle.dump(name_encodings, f)

class ZedImage(Node):
    def __init__(self):
        super().__init__('image__node')

        # Create a subscriber to the image topic
        self.subscription_obj = self.create_subscription(
            ObjectsStamped,
            '/zed_' + os.getenv('cam_loc') + '/zed_node_' + os.getenv('cam_loc') + '/body_trk/skeletons',  # Replace with the actual image topic name
            self.obj_callback,
            10)

        self.subscription_img = self.create_subscription(
            Image,
            '/zed_' + os.getenv('cam_loc') + '/zed_node_' + os.getenv('cam_loc') + '/left_raw/image_raw_color',  # Replace with the actual image topic name
            self.image_callback,
            10)

        # Create a publisher for the image count
        self.publisher = self.create_publisher(Int64, 'image_count', 10)
        self.bridge = CvBridge()
        self.cv2_image = None

    def scale_coordinates(self, original_size, new_size, coordinates):
    # original_size and new_size are tuples (width, height)
    # coordinates is a tuple (x, y)

        # Calculate scaling factors
        scale_x = new_size[0] / original_size[0]
        scale_y = new_size[1] / original_size[1]

        # Scale the coordinates
        scaled_x = int(coordinates[0] * scale_x)
        scaled_y = int(coordinates[1] * scale_y)

        return scaled_x, scaled_y
    def extract_head_bounding_box(self, image, box_coordinates):
        key_coordinates = [(int(x), int(y)) for x, y in box_coordinates]

        top_left_old = key_coordinates[0]
        bottom_right_old = key_coordinates[2]

        original_size = (720, 1080)  # Original image size (width, height)
        new_size = (image.shape[0], image.shape[1])

        top_left = self.scale_coordinates(original_size, new_size, top_left_old)
        bottom_right = self.scale_coordinates(original_size, new_size, bottom_right_old)

        top_left_x = top_left[0] - 10
        if top_left_x < 0:
            top_left_x = 0

        top_left_y = top_left[1] - 70
        if top_left_y < 0:
            top_left_y = 0
        #top_left_y = 0
        head_region = image[top_left_y:bottom_right[1], top_left_x:bottom_right[0]]
        return head_region

    def image_callback(self, image_msg):
        """Callback function that is called whenever a new image message is received."""
        self.cv2_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='passthrough')  # Preserve original encoding

    def obj_callback(self, pos_msg):
        if pos_msg.objects:
            key_pts = pos_msg.objects[0].bounding_box_2d.corners
            print(key_pts)
            # Extracting x, y coordinates for each Keypoint2Df
            box_coordinates = [(point.kp[0], point.kp[1]) for point in key_pts]
            print(box_coordinates)

            if self.cv2_image is not None:
                head_region = self.extract_head_bounding_box(self.cv2_image, box_coordinates)
                cv2.imshow('Head Region', head_region)
                cv2.waitKey(100)

def main(args=None):
    rclpy.init(args=args)
    node = ZedImage()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()

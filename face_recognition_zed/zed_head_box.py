import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Int64
from zed_interfaces.msg import ObjectsStamped
from zed_interfaces.msg import BoundingBox3D
from particle_filter_msgs.msg import PoseMsg
import cv2
from cv_bridge import CvBridge
import os
from pathlib import Path
from rclpy.callback_groups import ReentrantCallbackGroup
import face_recognition
import pickle
from std_msgs.msg import Float32MultiArray
from ament_index_python.packages import get_package_share_directory

import numpy as np
from collections import Counter

DEFAULT_ENCODINGS_PATH = os.getenv('encoding_loc') + "face_recognition_zed/output/encodings_.pkl"


# Path("../training").mkdir(exist_ok=True)
# Path("../output").mkdir(exist_ok=True)
# Path("../validation").mkdir(exist_ok=True)


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
        topic_body = '/zed_' + os.getenv('cam_loc') + '/zed_node_' + os.getenv('cam_loc') + '/body_trk/skeletons'
        print(topic_body)
        self.subscription_obj = self.create_subscription(
            ObjectsStamped,
            topic_body,
            # Replace with the actual image topic name
            self.obj_callback,
            10, callback_group=ReentrantCallbackGroup())
        topic_image = '/zed_' + os.getenv('cam_loc') + '/zed_node_' + os.getenv('cam_loc') + '/left_raw/image_raw_color'
        print(topic_image)
        self.subscription_img = self.create_subscription(
            Image,
            topic_image,
            # Replace with the actual image topic name
            self.image_callback,
            10, callback_group=ReentrantCallbackGroup())

        self.bridge = CvBridge()
        self.cv2_image = None
        self.model = "hog"
        self.encodings_location = os.getenv('encoding_loc') + "face_recognition_zed/output/encodings_.pkl"
        timer_period = 2  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        # print("self.encodings_location scsc  ",get_package_share_directory("face_recognition_zed"))
        print("self.encodings_location", self.encodings_location)
        self.tracked_label = None
        self.pose_pub = self.create_publisher(PoseMsg, os.getenv('cam_loc') + "_person_pose", 10)

    def recognize_faces(self, image_,
                        model: str = "hog"
                        ):

        with Path(self.encodings_location).open(mode="rb") as f:
            loaded_encodings = pickle.load(f)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        # print("Before rgb")
        rgb_frame = cv2.cvtColor(image_, cv2.COLOR_BGR2RGB)
        # print("After rgb")

        input_face_locations = face_recognition.face_locations(
            rgb_frame, model=model
        )
        input_face_encodings = face_recognition.face_encodings(
            rgb_frame, input_face_locations
        )

        name = None

        for bounding_box, unknown_encoding in zip(
                input_face_locations, input_face_encodings
        ):
            name = self._recognize_face(unknown_encoding, loaded_encodings)

            if not name:
                name = "unknown"

        #     # Removed print(name, bounding_box)
        #     top, right, bottom, left = bounding_box
        #     # Draw a box around the face
        #
        #     cv2.rectangle(rgb_frame, (left, top), (right, bottom), (0, 0, 255), 2)
        #
        #     # Draw a label with a name below the face
        #     cv2.rectangle(rgb_frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        #     font = cv2.FONT_HERSHEY_DUPLEX
        #     cv2.putText(rgb_frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        # #
        # cv2.imshow("Recognized Faces", rgb_frame)
        # cv2.waitKey(500)
        # cv2.destroyAllWindows()
        return name

    def _recognize_face(self, unknown_encoding, loaded_encodings):
        try:
            boolean_matches = face_recognition.compare_faces(
                loaded_encodings["encodings"], unknown_encoding
            )
            votes = Counter(
                name
                for match, name in zip(boolean_matches, loaded_encodings["names"])
                if match
            )
            if votes:
                return votes.most_common(1)[0][0]
        except Exception as e:
            print("!!!! Error in _recognize_face !!!")
            print(e)

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

        top_left_x = top_left[0] - 70
        if top_left_x < 0:
            top_left_x = 0

        top_left_y = top_left[1] - 70
        if top_left_y < 0:
            top_left_y = 0
        # top_left_y = 0
        head_region = image[top_left_y:bottom_right[1], top_left_x:bottom_right[0]]
        return head_region

    def image_callback(self, image_msg):
        try:
            # print("****************      Image      ************")
            if image_msg:
                """Callback function that is called whenever a new image message is received."""
                self.cv2_image = self.bridge.imgmsg_to_cv2(image_msg,
                                                           desired_encoding='passthrough')  # Preserve original encoding

        except Exception as e:
            print("!!!! Error in image_callback !!!")
            print(e)

    def obj_callback(self, pos_msg):
        # print("&&&&&&&&&&&&&&&&&&&& In Obj Callback &&&&&&&&&&&&&&&&&")
        self.objs = pos_msg.objects

    def timer_callback(self):
        print("Timer Callback")
        try:
            if self.objs:
                if self.tracked_label:
                    tracked_object = self.get_tracked_object(self.objs)
                    # print(tracked_object)
                    if tracked_object is not None:
                        tracked_pose = PoseMsg()
                        tracked_pose.label = "sajay"
                        tracked_pose.bounding_box = tracked_object.bounding_box_3d
                        # print("tracked_pose.bounding_box",tracked_pose.bounding_box)
                        print("tracked_object.dimensions_3d", tracked_object.dimensions_3d)

                        tracked_pose.dimensions_3d = Float32MultiArray(data=[tracked_object.dimensions_3d[0], tracked_object.dimensions_3d[1],
                                                      tracked_object.dimensions_3d[2]])
                        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$tracked_pose.dimensions_3d", tracked_pose.dimensions_3d)
                        self.pose_pub.publish(tracked_pose)
                    return

                exists = False
                for obj in self.objs:
                    key_pts = obj.bounding_box_2d.corners
                    # Extracting x, y coordinates for each Keypoint2Df
                    box_coordinates = [(point.kp[0], point.kp[1]) for point in key_pts]

                    if self.cv2_image is not None:
                        head_region = self.extract_head_bounding_box(self.cv2_image, box_coordinates)
                        # print('head_region.shape', head_region.shape)
                        name = self.recognize_faces(head_region, model=self.model)
                        print(name, "name2")

                        if name == "sajay":
                            exists = True
                            print("Label ID: ", obj.label_id)
                            self.tracked_label = obj.label_id
                if not exists:
                    tracked_pose = PoseMsg()
                    tracked_pose.label = ""
                    tracked_pose.bounding_box = BoundingBox3D()
                    tracked_pose.dimensions_3d = Float32MultiArray(data=[0.0, 0.0, 0.0])
                    self.pose_pub.publish(tracked_pose)

        except Exception as e:
            print("!!!! Error in obj_callback !!!")
            print(e)

    def get_tracked_object(self, objects):
        # tracked = any(d.label_id == self.tracked_label for d in objects)
        tracked_obj = next((d for d in objects if d.label_id == self.tracked_label), None)
        is_tracked = tracked_obj is not None

        if not is_tracked:
            self.tracked_label = None
        else:
            print("****** Is Tracked: ", is_tracked, " ******")

        return tracked_obj


def main(args=None):
    try:
        rclpy.init(args=args)
        node = ZedImage()
        exe = rclpy.executors.MultiThreadedExecutor()
        exe.add_node(node)
        while True:
            exe.spin_once(timeout_sec=1.0)
        node.destroy_node()
        rclpy.shutdown()
    except Exception as e:
        print("!!!! Error in main !!!")
        print(e)


if __name__ == '__main__':
    main()

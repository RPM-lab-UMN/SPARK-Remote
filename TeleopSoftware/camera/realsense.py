import pyrealsense2 as rs
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge
import numpy as np
import sys

class RealsenseNode(Node):
    def __init__(self):
        super().__init__('RealsenseNode')
        self.bridge = CvBridge()
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.working = False
        self.num_frames = 1
        self.align = rs.align(rs.stream.color)

        # Parse arguments
        if len(sys.argv) < 2:
            self.camera = None
        else:
            self.camera = sys.argv[1]

        # Camera configurations
        self.cameras = {
            '/lightning/wrist/': {
                'serial': '126122270307', 
                'hz': 20,
                'RGB': (480, 270, rs.format.rgb8, 30),
                'depth': None
            },
            '/thunder/wrist/': {
                'serial': '126122270722',
                'hz': 20,
                'RGB': (480, 270, rs.format.rgb8, 30),
                'depth': None
            },
            '/both/front/': {
                'serial': 'f1371786',
                'hz': 5,
                'RGB': (1920, 1080, rs.format.rgb8, 6),
                'depth': None
            },
            '/both/top/': {
                'serial': 'f1371463',
                'hz': 5,
                'RGB': (1920, 1080, rs.format.rgb8, 6),
                'depth': None
            },
        }

        # Set up publishers
        if self.camera is not None:
            if self.camera not in self.cameras:
                raise Exception(f"Camera {self.camera} not found")
            self.hz = self.cameras[self.camera]['hz']
            camera_topic = self.camera
            if self.camera[-1] == '/':
                camera_topic = self.camera[:-1]
            self.RGB_topic = f"/cameras/rgb{camera_topic}"
            self.depth_topic = f"/cameras/depth{camera_topic}"

            print(f"Publishing to {self.RGB_topic} and {self.depth_topic}")
            self.pub_rgb = self.create_publisher(Image, self.RGB_topic, 10)
            self.pub_depth = self.create_publisher(Image, self.depth_topic, 10)
        else:
            self.hz = 10

        # Configure the camera
        self.configure_camera()

        # Start the pipeline
        self.pipeline.start(self.config)

        # Create a timer to process frames
        self.timer = self.create_timer(1.0 / self.hz, self.process_frames)

    def configure_camera(self):
        if self.camera is not None:
            camera_config = self.cameras[self.camera]
            self.config.enable_device(camera_config['serial'])
            if camera_config['RGB'] is not None:
                self.get_logger().info(f"RGB ({self.RGB_topic}): {camera_config['RGB']}")
                self.config.enable_stream(rs.stream.color, *camera_config['RGB'])
                self.mode = "RGB_"
            if camera_config['depth'] is not None:
                self.get_logger().info(f"DEPTH ({self.depth_topic}): {camera_config['depth']}")
                self.config.enable_stream(rs.stream.depth, *camera_config['depth'])
                self.mode += "DEPTH_"
        else:
            self.mode = "RGB_DEPTH_"

    def process_frames(self):
        try:
            frames = self.pipeline.wait_for_frames()
            if self.mode == "RGB_DEPTH_":
                aligned_frames = self.align.process(frames)
                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()
                if not depth_frame or not color_frame:
                    self.get_logger().warn("Missing frames")
                    return
            elif self.mode == "RGB_":
                color_frame = frames.get_color_frame()
                depth_frame = None
            elif self.mode == "DEPTH_":
                color_frame = None
                depth_frame = frames.get_depth_frame()

            if depth_frame is not None:
                depth_image = np.asanyarray(depth_frame.get_data())
                msg = self.bridge.cv2_to_imgmsg(depth_image, encoding="16UC1")
                self.pub_depth.publish(msg)

            if color_frame is not None:
                color_image = np.asanyarray(color_frame.get_data())
                msg = self.bridge.cv2_to_imgmsg(color_image, encoding="bgr8")
                self.pub_rgb.publish(msg)

            if (depth_frame is not None or color_frame is not None) and not self.working:
                self.working = True
                self.get_logger().info(f"Publishing {self.mode} frames: {self.camera}")

            if self.num_frames % (self.hz * 20) == 0:
                self.get_logger().info(f"{self.camera} frames: {self.num_frames}")
            self.num_frames += 1

        except Exception as e:
            self.get_logger().error(f"Error processing frames: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = RealsenseNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.pipeline.stop()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
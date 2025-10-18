import pyrealsense2 as rs
import numpy as np
import cv2

class RealSenseCamera:
    def __init__(self, serial_number=None, width=640, height=480, fps=30):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        if serial_number:
            self.config.enable_device(serial_number)
        self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self.pipeline.start(self.config)
        self.align = rs.align(rs.stream.color)
        print("RealSense camera initialized.")

    def get_frames(self):
        """get aligned color and depth frames"""
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)

        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            return None, None

        # Convert to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        return color_image, depth_image

    def stop(self):
        self.pipeline.stop()
        print("Camera pipeline stopped.")


if __name__ == "__main__":
    # Replace with your actual serial number
    cam = RealSenseCamera(serial_number='128422270284')
    print("Press 'q' to quit.")

    try:
        while True:
            color_frame, depth_frame = cam.get_frames()
            if color_frame is not None and depth_frame is not None:
                # Normalize depth map for display
                depth_colormap = cv2.applyColorMap(
                    cv2.convertScaleAbs(depth_frame, alpha=0.03),
                    cv2.COLORMAP_JET
                )

                # Display color and depth frames
                cv2.imshow("Color Frame", color_frame)
                cv2.imshow("Depth Frame", depth_colormap)


                key = cv2.waitKey(1) & 0xFF
                if key == ord('s'):
                    # save color and depth images
                    cv2.imwrite(f"color.png", color_frame)
                    cv2.imwrite(f"depth.png", depth_colormap)
                    depth_image = cv2.convertScaleAbs(depth_frame, alpha=0.03)
                    print("Depth range:", depth_image.min(), "→", depth_image.max())
                    print(f"Saved color.png and depth.png")

                elif key == ord('q'):
                    break
    finally:
        cam.stop()
        cv2.destroyAllWindows()
        print("Camera stopped.")

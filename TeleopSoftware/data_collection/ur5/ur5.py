import numpy as np
import rtde_control
import rtde_receive
from ur5.gripper import RobotiqGripper

DEFAULT_LIGHTNING_IP = '192.168.0.101'
DEFAULT_THUNDER_IP = '192.168.0.102'

LIGHTNING_HOME = [-3.092430591583252,
                  -2.535433530807495,
                  -1.2771631479263306,
                  -1.0458279848098755,
                  -0.0320628322660923,
                  -0.025522056967020035]
THUNDER_HOME   = [3.157623052597046, -0.5073397916606446, 0.9275072256671351, -2.031027456323141, 0.02100839652121067, 0.16949762403964996]

SPEED = 0.5
ACCELERATION = 0.5
DT = 0.1
LOOKAHEAD_TIME = 0.2
GAIN = 500

class RobotController:
    def __init__(self, arm: str, robot_ip: str, need_control: bool = False, need_gripper: bool = False):
        self._ip = robot_ip if (arm in ["thunder", "lightning"]) and (robot_ip is not None) else DEFAULT_LIGHTNING_IP
        self.home = THUNDER_HOME if arm == 'thunder' else LIGHTNING_HOME
        self.gripper = self._init_gripper() if need_gripper else None
        self.receiver = rtde_receive.RTDEReceiveInterface(self._ip)
        self.controller = rtde_control.RTDEControlInterface(self._ip) if need_control else None

    def _init_gripper(self) -> RobotiqGripper:
        gripper = RobotiqGripper()
        gripper.connect(self._ip, 63352)
        gripper.activate()
        gripper.set_enable(True)
        return gripper

    # ----------------------------
    # Robot State
    # ----------------------------
    def get_eff_pose(self) -> list[float]:
        """Get current TCP pose [x,y,z,Rx,Ry,Rz]."""
        return self.receiver.getActualTCPPose()

    def get_joint_angles(self) -> list[float]:
        """Get current joint angles [rad]."""
        return self.receiver.getActualQ()

    def get_tcp_force(self) -> list[float]:
        """Get wrench (forces/torques) at TCP [Fx,Fy,Fz,Tx,Ty,Tz]."""
        return self.receiver.getActualTCPForce()

    # ----------------------------
    # Motion Control
    # ----------------------------
    def moveJ(self, joints: list[float], speed=SPEED, accel=ACCELERATION, async_flag: bool = False):
        """Move in joint space."""
        self.controller.moveJ(joints, speed, accel, async_flag)

    def moveL(self, pose: list[float], speed=SPEED, accel=ACCELERATION, async_flag: bool = False):
        """Move in Cartesian space (linear)."""
        self.controller.moveL(pose, speed, accel, async_flag)

    def servoJ(self, joints: list[float], speed=SPEED, accel=ACCELERATION):
        """Servo motion in joint space (smooth real-time)."""
        self.controller.servoJ(joints, speed, accel, DT, LOOKAHEAD_TIME, GAIN)

    def servoL(self, pose: list[float], speed=SPEED, accel=ACCELERATION):
        """Servo motion in Cartesian space (smooth real-time)."""
        self.controller.servoL(pose, speed, accel, DT, LOOKAHEAD_TIME, GAIN)

    def stop(self):
        """Stop robot immediately."""
        self.controller.stopJ(ACCELERATION)

    def freeDrive(self):
        """Enable hand-guiding mode."""
        self.controller.teachMode()
        try:
            while True:
                user_input = input("Enter 'DONE' to Exit Free Drive Mode: ")
                if user_input.strip().upper() == "DONE":
                    break
        finally:
            self.controller.endTeachMode()

    def go_home(self):
        """Move robot to predefined home pose."""
        print(f"Moving to {self.home}")
        self.controller.moveJ(self.home, SPEED, ACCELERATION, False)

    # ----------------------------
    # I/O Functions
    # ----------------------------
    def set_digital_out(self, pin: int, value: bool):
        """Set digital output pin (0-7)."""
        self.controller.setStandardDigitalOut(pin, value)

    # ----------------------------
    # Gripper
    # ----------------------------
    def gripper_close(self, value=255):
        if self.gripper:
            self.gripper.set(int(value))

    def gripper_open(self, value=0):
        if self.gripper:
            self.gripper.set(int(value))

    def get_current_position(self) -> int:
        """Returns the current position as returned by the physical hardware."""
        return self.gripper._get_var(self.gripper.POS)

if __name__ == "__main__":
    robot = RobotController('thunder', need_control=True, need_gripper=False)
    print("Current pose:", robot.get_eff_pose())
    print("Current joints:", robot.get_joint_angles())
    robot.go_home()

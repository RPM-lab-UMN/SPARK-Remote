import rtde_control
import rtde_receive
from UR.dashboard import rtde_dashboard
from UR.gripper import RobotiqGripper
import time

class UR:
    def __init__(self, names, ip_addresses, enable_grippers):
        self.ur_dashboard = {}
        self.ur_control = {}
        self.ur_receive = {}
        self.ur_grippers = {}
        self.names = names
        self.enable_grippers = enable_grippers
        self.ips = {}
        for name, ip in zip(names, ip_addresses):
            self.ips[name] = ip
        self.mode = {}

    def zeroFtSensor(self, name):
        self.ur_control[name].zeroFtSensor()

    def get_dashboard(self, name):
        return self.ur_dashboard[name]

    # def get_control(self, name):
    #     return self.ur_control[name]
    
    def get_receive(self, name):
        return self.ur_receive[name]
    
    def get_gripper(self, name):
        return self.ur_grippers[name]
    
    def stop(self, name):
        if name in self.mode:
            if self.mode[name] == 'freedrive':
                self.ur_control[name].endFreedriveMode()
            elif self.mode[name] == 'servoL' or self.mode[name] == 'servoJ':
                self.ur_control[name].servoStop()
            elif self.mode[name] == 'moveJ':
                self.ur_control[name].stopJ()
            elif self.mode[name] == 'forceMode':
                self.ur_control[name].forceModeStop()
            elif self.mode[name] == 'moveL':
                self.ur_control[name].stopL() 
            elif self.mode[name] == 'speedJ':
                self.ur_control[name].speedStop()
            elif self.mode[name] == 'speedL':
                self.ur_control[name].speedStop()
        self.mode[name] = None

    def servoL(self, name, perams):
        # print(f"ServoL: {perams}")
        if self.mode[name] != 'servoL':
            self.stop(name)
            self.mode[name] = 'servoL'
        self.ur_control[name].servoL(perams[0], perams[1], perams[2], perams[3], perams[4], perams[5])

    def servoJ(self, name, perams):
        # print(f"ServoJ: {perams}")
        if self.mode[name] != 'servoJ':
            self.stop(name)
            self.mode[name] = 'servoJ'
        self.ur_control[name].servoJ(perams[0], perams[1], perams[2], perams[3], perams[4], perams[5])

    def moveJ(self, name, perams):
        # print(f"MoveJ: {perams}")
        if self.mode[name] != 'moveJ':
            self.stop(name)
            self.mode[name] = 'moveJ'
        self.ur_control[name].moveJ(perams[0], perams[1], perams[2])

    def moveL(self, name, perams):
        # print(f"MoveL: {perams}")
        if self.mode[name] != 'moveL':
            self.stop(name)
            self.mode[name] = 'moveL'
        self.ur_control[name].moveL(perams[0], perams[1], perams[2], perams[3])

    def speedJ(self, name, perams):
        # print(f"SpeedJ: {perams}")
        if self.mode[name] != 'speedJ':
            self.stop(name)
            self.mode[name] = 'speedJ'
        self.ur_control[name].speedJ(perams[0], perams[1], perams[2])

    def speedL(self, name, perams):
        # print(f"SpeedL: {perams}")
        if self.mode[name] != 'speedL':
            self.stop(name)
            self.mode[name] = 'speedL'
        self.ur_control[name].speedL(perams[0], perams[1], perams[2])

    def forceMode(self, name, perams):
        if self.mode[name] != 'forceMode':
            # print(f"ForceMode: {perams}")
            self.stop(name)
            self.mode[name] = 'forceMode'
            self.ur_control[name].forceMode(perams[0], perams[1], perams[2], perams[3], perams[4])
        # Params: task_frame, selection_vector, wrench_up, force_type, limits
    
    def freeDrive(self, name, enable):
        # print(f"FreeDrive: {enable}")
        if self.mode[name] != 'freedrive':
            self.stop(name)
        if enable:
            self.ur_control[name].freedriveMode()
            self.mode[name] = 'freedrive'
        else:
            self.ur_control[name].endFreedriveMode()
            self.mode[name] = None

    def getActualQ(self, name):
        return self.ur_receive[name].getActualQ()
    
    def getJointTorques(self, name):
        return self.ur_control[name].getJointTorques()

    def triggerProtectiveStop(self, name):
        self.ur_control[name].triggerProtectiveStop()
        self.mode[name] = None

    def init_dashboard(self, name):
        try:
            if name in self.ur_dashboard:
                del self.ur_dashboard[name]
            self.ur_dashboard[name] = rtde_dashboard(self.ips[name])
            print(f"Connected to {name} dashboard at {self.ips[name]}")
        except RuntimeError as e:
            print(f"\tFailed to connect to {name} dashboard at {self.ips[name]}")
            print("\t"+str(e))
            return False
        return True

    def init_arm(self, name, count=0, enable_control=True):
        # print(f"Enabling control: {enable_control}")
        try:
            self.mode[name] = None
            if enable_control[name]:
                print(f"Enabling control: {name}")
                if name in self.ur_control:
                    self.ur_control[name].disconnect()
                    self.ur_control[name].reconnect()
                else:
                    self.ur_control[name] = rtde_control.RTDEControlInterface(self.ips[name], 500)

            if name in self.ur_receive:
                self.ur_receive[name].disconnect()
                self.ur_receive[name].reconnect()
            self.ur_receive[name] = rtde_receive.RTDEReceiveInterface(self.ips[name])

            print(f"Connected to {name} at {self.ips[name]}")
        except RuntimeError as e:
            print(f"\tFailed to connect to {name} at {self.ips[name]}")
            print("\t"+str(e))
            if count < 5:
                return self.init_arm(name, count+1, enable_control=enable_control)
            else:
                return False
        try:
            if name not in self.ur_grippers:
                self.ur_grippers[name] = RobotiqGripper()
                self.ur_grippers[name].connect(self.ips[name], 63352)
                if self.enable_grippers[name]:
                    self.ur_grippers[name].activate()
                    print(f"Connected to {name} gripper")
                self.ur_grippers[name].set_enable(True)
        except RuntimeError as e:
            print(f"\tFailed to connect to {name} gripper")
            print("\t"+str(e))
        return True

    def __del__(self):
        for name in self.names:
            if name in self.ur_dashboard:
                del self.ur_dashboard[name]
            if name in self.ur_control:
                self.ur_control[name].disconnect()
                del self.ur_control[name]
            if name in self.ur_receive:
                self.ur_receive[name].disconnect()
                del self.ur_receive[name]
            if name in self.ur_grippers:
                del self.ur_grippers[name]
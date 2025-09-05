import time
import os
import sys
from pymycobot.mycobot280 import MyCobot280
from pymycobot.genre import Angle, Coord

sys.path.append(os.path.dirname(__file__))
from port_setup import setup

reset = [0, 0, 0, 0, 0, 0]


def test(mycobot):
    print("\nStart check basic options\n")

    angles = [0, 0, 0, 0, 0, 0]
    mycobot.send_angles(angles, 100)

    print("::send_angles() ==> angles {}, speed 100\n".format(angles))
    time.sleep(1)
    print("Initial coords {}\n".format(mycobot.get_coords()))
    print("::get_angles() ==> degrees: {}\n".format(mycobot.get_angles()))

    print("::get_coords() ==> coords {}\n".format(mycobot.get_coords()))
    angles = [0, -100, 50, 50, 0, 35]
    mycobot.send_angles(angles, 100)
    mycobot.set_gripper_state(0, 100)
    time.sleep(3)

    mycobot.send_angle(Angle.J1.value, -45, 80)
    print("::send_angle() ==> angle: joint1, degree: 90, speed: 50\n")
    time.sleep(2)

    mycobot.send_angle(Angle.J1.value, 0, 80)
    print("::send_angle() ==> angle: joint1, degree: 90, speed: 50\n")
    time.sleep(2)

    mycobot.send_angle(Angle.J1.value, 60, 80)
    print("::send_angle() ==> angle: joint1, degree: 90, speed: 50\n")
    time.sleep(2)

    mycobot.send_angle(Angle.J1.value, 0, 80)
    print("::send_angle() ==> angle: joint1, degree: 90, speed: 50\n")
    time.sleep(2)
    
    mycobot.send_angle(Angle.J1.value, -45, 80)
    print("::send_angle() ==> angle: joint1, degree: 90, speed: 50\n")
    time.sleep(2)

    mycobot.send_angle(Angle.J1.value, 0, 80)
    print("::send_angle() ==> angle: joint1, degree: 90, speed: 50\n")
    time.sleep(2)

    mycobot.send_angle(Angle.J1.value, 60, 80)
    print("::send_angle() ==> angle: joint1, degree: 90, speed: 50\n")
    time.sleep(2)

    mycobot.send_angle(Angle.J1.value, 0, 80)
    print("::send_angle() ==> angle: joint1, degree: 90, speed: 50\n")
    time.sleep(2)
    
    angles = [0, 0, 0, 0, 0, 0]
    mycobot.send_angles(angles, 100)
    mycobot.set_gripper_state(1, 100)

    print("::send_angles() ==> angles {}, speed 100\n".format(angles))
    time.sleep(1)

    print("=== check end ===\n")


if __name__ == "__main__":
    print(
        """
--------------------------------------------
| This file will test basic option method: |
|     set_led_color()                      |
|     send_angles()                        |
|     get_angles()                         |
|     send_angle()                         |
|     send_radians()                       |
|     get_radians()                        |
|     send_coords()                        |
|     get_coords()                         |
|     send_coord()                         |
--------------------------------------------
          """
    )
    time.sleep(3)
    # port = subprocess.check_output(['echo -n /dev/ttyUSB*'],
    # shell=True).decode()
    # with open(os.path.dirname(__file__) + "/port.txt") as f:
        # port = f.read().strip().replace("\n", "")
        # print(port)
    mycobot = setup()
    test(mycobot)

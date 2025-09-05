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
    mycobot.set_gripper_state(1, 100)
    print("::send_angles() ==> angles {}, speed 100\n".format(angles))
    time.sleep(1)
    print("Initial coords {}\n".format(mycobot.get_coords()))
    print("::get_angles() ==> degrees: {}\n".format(mycobot.get_angles()))

    mycobot.send_angle(Angle.J2.value, -65, 50)
    print("::send_angle() ==> angle: joint1, degree: 90, speed: 50\n")
    time.sleep(1)

    mycobot.send_angle(Angle.J6.value, -130, 50)
    print("::send_angle() ==> angle: joint1, degree: 90, speed: 50\n")
    time.sleep(2)
    

    mycobot.set_gripper_state(0, 100)
    print("::set_gripper_state() ==> set gripper state 0, speed 70\n")
    time.sleep(2)

    mycobot.send_angle(Angle.J2.value, -100, 50)
    print("::send_angle() ==> angle: joint1, degree: 90, speed: 50\n")
    time.sleep(2)

    flag = mycobot.is_gripper_moving()
    print("Is gripper moving: {}".format(flag))
    time.sleep(1)

    # Set the current position to (2048).
    # Use it when you are sure you need it.
    # Gripper has been initialized for a long time. Generally, there
    # is no need to change the method.
    # mc.set_gripper_ini()

    # mycobot.set_encoder(7, 2048, 50)
    # time.sleep(3)
    # mycobot.set_encoder(7, 1300, 50)
    # time.sleep(3)

    
    mycobot.set_gripper_state(1, 100)
    print("::set_gripper_state() ==> set gripper state 1, speed 70\n")
    time.sleep(3)

    print("")
    print(mycobot.get_gripper_value())
    # mycobot.send_coord(Coord.X.value, -40, 70)
    # print("::send_coord() ==> send coord id: X, coord value: -40, speed: 70\n")
    # time.sleep(2)

    # print("::set_free_mode()\n")
    mycobot.send_angles(reset, 100)
    
    print("::set_gripper_state() ==> set gripper state 1, speed 70\n")
    # time.sleep(2)
    # mycobot.set_gripper_state(0, 100)
    mycobot.send_angle(Angle.J6.value, 0, 100)
    time.sleep(2)
    mycobot.send_angle(Angle.J6.value, 180, 100)
    time.sleep(2)
    mycobot.send_angle(Angle.J6.value, 0, 100)
    time.sleep(2)
    mycobot.send_angle(Angle.J6.value, 180, 100)
    # time.sleep(5)
    # mycobot.release_all_servos()

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

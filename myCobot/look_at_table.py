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
    mycobot.send_angle(Angle.J4.value, -45, 50)
    mycobot.send_angle(Angle.J6.value, 45, 50)
    print("=== check end ===\n")

    time.sleep(3)
    print("::get_coords() ==> coords {}\n".format(mycobot.get_coords()))


if __name__ == "__main__":
    mycobot = setup()
    test(mycobot)

import time
import os
import sys
from pymycobot.mycobot280 import MyCobot280
from pymycobot.genre import Angle, Coord

sys.path.append(os.path.dirname(__file__))
from port_setup import setup

def test(mycobot):
    print("\nStart check basic options\n")

    print("::get_angles() ==> degrees: {}\n".format(mycobot.get_angles()))
    print("::get_coords() ==> coords {}\n".format(mycobot.get_coords()))
    
    print("=== check end ===\n")


if __name__ == "__main__":

    # time.sleep(3)
    # port = subprocess.check_output(['echo -n /dev/ttyUSB*'],
    # shell=True).decode()
    # with open(os.path.dirname(__file__) + "/port.txt") as f:
        # port = f.read().strip().replace("\n", "")
        # print(port)
    mycobot = setup()
    test(mycobot)

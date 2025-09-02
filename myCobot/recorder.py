import json
import time
import serial
import serial.tools.list_ports
from pymycobot.mycobot280 import MyCobot280

def setup():
    """Detects and connects to the MyCobot280."""
    print("Detecting serial ports...")
    plist = list(serial.tools.list_ports.comports())
    if not plist:
        print("Error: No serial ports found.")
        exit()

    # Automatically select the first detected port
    port_info = plist[0]
    port = port_info.device
    baud = 115200
    
    print(f"Connecting to port: {port}, baud: {baud}")
    mc = MyCobot280(port, baud)
    return mc


# --- Main execution ---
if __name__ == '__main__':
    # 1. Connect to the robot
    mc = setup()
    time.sleep(2)
    
    # 2. Release motors for manual movement
    print("\nRobot is ready. Releasing motors for manual movement.")
    mc.release_all_servos()
    time.sleep(1)

    # 3. Prepare to record
    recorded_coords = []
    print("\nInstructions:")
    print(" - Manually move the robot to a desired position.")
    
    try:
        # 4. Main loop to get user input
        while True:
            # Prompt user for action
            user_input = input(" > Type 'r' to record, or 's' to save and exit: ").lower()

            if user_input == 'r':
                # Record coordinates
                coords = mc.get_coords()
                if coords:
                    recorded_coords.append(coords)
                    print(f"  Coordinates recorded: {coords}")
            
            elif user_input == 's':
                # Save and exit
                with open('recorded_coords_280.json', 'w') as f:
                    json.dump(recorded_coords, f, indent=4)
                print(f"\nSuccessfully saved {len(recorded_coords)} coordinates to 'recorded_coords_280.json'.")
                break
            
            else:
                print("  Invalid input. Please try again.")

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # 5. Lock the robot's motors before exiting
        print("Locking robot motors.")
        mc.power_on()
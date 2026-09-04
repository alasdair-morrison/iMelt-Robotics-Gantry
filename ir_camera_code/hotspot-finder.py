import os
import sys
import threading
print(sys.executable)
import PySpin
import time
import matplotlib
matplotlib.use('Qt5Agg')  # Add this line to force an interactive window
import matplotlib.pyplot as plt
import keyboard
import numpy as np
import thermal_analysis as ta
from zaber_motion import Units
from zaber_motion.ascii import Connection

CONTINUE_RECORDING = True


def gantryControl(x, y, z):
    """
    This function controls the gantry to move to points entered by the user and runs on a 
    separate thread to allow the GUI to run in parallel.
    """
    global CONTINUE_RECORDING
    while CONTINUE_RECORDING:
        inputString= input("Enter the coordinates to move the gantry to (x, y, z) or type 'exit' to quit: ")
        if inputString == 'exit':
            print("Exiting gantry control thread.")
            break
        inputs = inputString.split(',')
        if len(inputs) != 3:
            if len(inputs) == 1:
                targetX = float(inputs[0])
                if not (0 <= targetX <= 249):
                    print("Invalid input. Please enter coordinates within the valid range.")
                    continue
                else: 
                    x.move_absolute(targetX, Units.LENGTH_MILLIMETRES, wait_until_idle=False)
                    x.wait_until_idle()
                    continue
            elif len(inputs) == 2:
                if inputs[0] == '':
                    targetX = 0.0
                    targetY = float(inputs[1])
                    if not (0 <= targetY <= 99):
                        print("Invalid input. Please enter coordinates within the valid range.")
                        continue
                    else:
                        y.move_absolute(targetY, Units.LENGTH_MILLIMETRES, wait_until_idle=False)
                        y.wait_until_idle()
                        continue
                targetX = float(inputs[0])
                targetY = float(inputs[1])
                if not (0 <= targetX <= 249 and 0 <= targetY <= 99):
                    print("Invalid input. Please enter coordinates within the valid range.")
                    continue
                else:
                    x.move_absolute(targetX, Units.LENGTH_MILLIMETRES, wait_until_idle=False)
                    y.move_absolute(targetY, Units.LENGTH_MILLIMETRES, wait_until_idle=False)
                    x.wait_until_idle()
                    y.wait_until_idle()
                    continue
            else:
                print("Invalid input. Please enter coordinates in the format 'x, y, z'.")
                continue
        else:
            try:
                if inputs[0] == '' and inputs[1] == '':
                    targetX = 0.0
                    targetY = 0.0
                    targetZ = float(inputs[2])
                elif inputs[1] == '':
                    targetX = float(inputs[0])
                    targetY = 0.0
                    targetZ = float(inputs[2])
                else:
                    targetX = float(inputs[0])
                    targetY = float(inputs[1])
                    targetZ = float(inputs[2])
            except ValueError:
                print("Invalid input. Please enter coordinates in the format 'x, y, z'.")
                continue
        if not (0 <= targetX <= 249 and 0 <= targetY <= 99 and 0 <= targetZ <= 249):
            print("Invalid input. Please enter coordinates within the valid range.")
            continue
        try:
            if targetX == 0.0 and targetY == 0.0:
                z.move_absolute(targetZ, Units.LENGTH_MILLIMETRES, wait_until_idle=False)
                z.wait_until_idle()
                continue
            elif targetX == 0.0:
                y.move_absolute(targetY, Units.LENGTH_MILLIMETRES, wait_until_idle=False)
                z.move_absolute(targetZ, Units.LENGTH_MILLIMETRES, wait_until_idle=False)
                y.wait_until_idle()
                z.wait_until_idle()
                continue
            elif targetY == 0.0:
                x.move_absolute(targetX, Units.LENGTH_MILLIMETRES, wait_until_idle=False)
                z.move_absolute(targetZ, Units.LENGTH_MILLIMETRES, wait_until_idle=False)
                x.wait_until_idle()
                z.wait_until_idle()
                continue
            x.move_absolute(targetX, Units.LENGTH_MILLIMETRES, wait_until_idle=False)
            y.move_absolute(targetY, Units.LENGTH_MILLIMETRES, wait_until_idle=False)
            z.move_absolute(targetZ, Units.LENGTH_MILLIMETRES, wait_until_idle=False)
            x.wait_until_idle()
            y.wait_until_idle()
            z.wait_until_idle()
        except ValueError:
            print("Invalid input. Please enter coordinates in the format 'x, y, z'.")

class IRFormatType:
    LINEAR_10MK = 1
    LINEAR_100MK = 2
    RADIOMETRIC = 3

CHOSEN_IR_TYPE = IRFormatType.RADIOMETRIC


def handle_close(evt):
    """
    This function will close the GUI when close event happens.

    :param evt: Event that occurs when the figure closes.
    :type evt: Event
    """
    global CONTINUE_RECORDING
    CONTINUE_RECORDING = False


def acquire_and_display_images(cam, nodemap, nodemap_tldevice):
    """
    This function continuously acquires images from a device and display them in a GUI.

    :param cam: Camera to acquire images from.
    :param nodemap: Device nodemap.
    :param nodemap_tldevice: Transport layer device nodemap.
    :type cam: CameraPtr
    :type nodemap: INodeMap
    :type nodemap_tldevice: INodeMap
    :return: True if successful, False otherwise.
    :rtype: bool
    """
    global CONTINUE_RECORDING

    sNodemap = cam.GetTLStreamNodeMap()
    node_bufferhandling_mode = PySpin.CEnumerationPtr(sNodemap.GetNode('StreamBufferHandlingMode'))
    node_pixel_format = PySpin.CEnumerationPtr(nodemap.GetNode('PixelFormat'))
    node_pixel_format_mono16 = PySpin.CEnumEntryPtr(node_pixel_format.GetEntryByName('Mono16'))
    pixel_format_mono16 = node_pixel_format_mono16.GetValue()
    node_pixel_format.SetIntValue(pixel_format_mono16)

    if CHOSEN_IR_TYPE == IRFormatType.LINEAR_10MK:
        node_IRFormat = PySpin.CEnumerationPtr(nodemap.GetNode('IRFormat'))
        node_temp_linear_high = PySpin.CEnumEntryPtr(node_IRFormat.GetEntryByName('TemperatureLinear10mK'))
        node_IRFormat.SetIntValue(node_temp_linear_high.GetValue())
    elif CHOSEN_IR_TYPE == IRFormatType.LINEAR_100MK:
        node_IRFormat = PySpin.CEnumerationPtr(nodemap.GetNode('IRFormat'))
        node_temp_linear_low = PySpin.CEnumEntryPtr(node_IRFormat.GetEntryByName('TemperatureLinear100mK'))
        node_IRFormat.SetIntValue(node_temp_linear_low.GetValue())
    elif CHOSEN_IR_TYPE == IRFormatType.RADIOMETRIC:
        node_IRFormat = PySpin.CEnumerationPtr(nodemap.GetNode('IRFormat'))
        node_temp_radiometric = PySpin.CEnumEntryPtr(node_IRFormat.GetEntryByName('Radiometric'))
        node_IRFormat.SetIntValue(node_temp_radiometric.GetValue())

    if not PySpin.IsAvailable(node_bufferhandling_mode) or not PySpin.IsWritable(node_bufferhandling_mode):
        print('Unable to set stream buffer handling mode.. Aborting...')
        return False

    node_newestonly = node_bufferhandling_mode.GetEntryByName('NewestOnly')
    if not PySpin.IsAvailable(node_newestonly) or not PySpin.IsReadable(node_newestonly):
        print('Unable to set stream buffer handling mode.. Aborting...')
        return False

    node_bufferhandling_mode.SetIntValue(node_newestonly.GetValue())

    print('*** IMAGE ACQUISITION ***\n')
    try:
        node_acquisition_mode = PySpin.CEnumerationPtr(nodemap.GetNode('AcquisitionMode'))
        if not PySpin.IsAvailable(node_acquisition_mode) or not PySpin.IsWritable(node_acquisition_mode):
            return False

        node_acquisition_mode_continuous = node_acquisition_mode.GetEntryByName('Continuous')
        if not PySpin.IsAvailable(node_acquisition_mode_continuous) or not PySpin.IsReadable(node_acquisition_mode_continuous):
            return False

        node_acquisition_mode.SetIntValue(node_acquisition_mode_continuous.GetValue())
        cam.BeginAcquisition()
        print('Acquiring images...')

        device_serial_number = ''
        node_device_serial_number = PySpin.CStringPtr(nodemap_tldevice.GetNode('DeviceSerialNumber'))
        if PySpin.IsAvailable(node_device_serial_number) and PySpin.IsReadable(node_device_serial_number):
            device_serial_number = node_device_serial_number.GetValue()
            print('Device serial number retrieved as %s...' % device_serial_number)

        # Retrieve Calibration constants
        R = PySpin.CFloatPtr(nodemap.GetNode('R')).GetValue()
        B = PySpin.CFloatPtr(nodemap.GetNode('B')).GetValue()
        F = PySpin.CFloatPtr(nodemap.GetNode('F')).GetValue()
        X = PySpin.CFloatPtr(nodemap.GetNode('X')).GetValue()
        A1 = PySpin.CFloatPtr(nodemap.GetNode('alpha1')).GetValue()
        A2 = PySpin.CFloatPtr(nodemap.GetNode('alpha2')).GetValue()
        B1 = PySpin.CFloatPtr(nodemap.GetNode('beta1')).GetValue()
        B2 = PySpin.CFloatPtr(nodemap.GetNode('beta2')).GetValue()
        J1 = PySpin.CFloatPtr(nodemap.GetNode('J1')).GetValue()
        J0 = PySpin.CIntegerPtr(nodemap.GetNode('J0')).GetValue()

        # Initialize interactive plot once outside the loop
        plt.ion()
        fig, ax = plt.subplots(figsize=(9, 7))
        fig.suptitle('A700 Temperature Radiometric', fontsize=12, weight='bold')
        fig.canvas.mpl_connect('close_event', handle_close)
        
        # Adjust subplot bottom margin so HUD text does not overlap with axes or colorbar
        fig.subplots_adjust(bottom=0.15)

        im_display = None
        cbar = None
        hot_marker, = ax.plot([], [], marker='+', color='red', markersize=16, markeredgewidth=2)

        # Visually distinct text overlays:
        # 1. Pixel/Relative coordinates (upper line, muted neutral styling)
        pixel_text = fig.text(0.5, 0.07, '', ha='center', va='center', fontsize=9, color='#555555')
        # 2. Real-World Gantry coordinates (bottom line, highlighted bold box)
        real_world_text = fig.text(
            0.5, 0.025, '', ha='center', va='center', fontsize=10, weight='bold', color='#004C99',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#EBF3FB', edgecolor='#B3D1FF', alpha=0.9)
        )

        if CHOSEN_IR_TYPE == IRFormatType.RADIOMETRIC:
            Emiss, TRefl, TAtm, Humidity = 0.97, 293.15, 293.15, 0.55
            Dist, ExtOpticsTransmission = 2, 1
            ExtOpticsTemp = TAtm
            TAtmC = TAtm - 273.15
            H2O = Humidity * np.exp(1.5587 + 0.06939 * TAtmC - 0.00027816 * TAtmC**2 + 0.00000068455 * TAtmC**3)
            Tau = X * np.exp(-np.sqrt(Dist) * (A1 + B1 * np.sqrt(H2O))) + (1 - X) * np.exp(-np.sqrt(Dist) * (A2 + B2 * np.sqrt(H2O)))
            r1 = ((1 - Emiss) / Emiss) * (R / (np.exp(B / TRefl) - F))
            r2 = ((1 - Tau) / (Emiss * Tau)) * (R / (np.exp(B / TAtm) - F))
            r3 = ((1 - ExtOpticsTransmission) / (Emiss * Tau * ExtOpticsTransmission)) * (R / (np.exp(B / ExtOpticsTemp) - F))
            K2 = r1 + r2 + r3

        background_Temp = ta.load_background(filename="background.npy") if os.path.exists("background.npy") else None
        transform_matrix = ta.load_transform_matrix("transform_matrix.json") if os.path.exists("transform_matrix.json") else None

        print('Press Q to stop streaming')

        while CONTINUE_RECORDING:
            try:
                image_result = cam.GetNextImage(1000)

                if image_result.IsIncomplete():
                    print('Image incomplete with status %d...' % image_result.GetImageStatus())
                else:
                    image_data = image_result.GetNDArray()

                    if CHOSEN_IR_TYPE == IRFormatType.LINEAR_10MK:
                        display_data = (image_data * 0.01) - 273.15
                    elif CHOSEN_IR_TYPE == IRFormatType.LINEAR_100MK:
                        display_data = (image_data * 0.1) - 273.15
                    elif CHOSEN_IR_TYPE == IRFormatType.RADIOMETRIC:
                        image_Radiance = (image_data - J0) / J1
                        image_Temp = (B / np.log(R / ((image_Radiance / Emiss / Tau) - K2) + F)) - 273.15
                        clean_temp_array = ta.subtract_background(image_Temp, background_Temp) if background_Temp is not None else image_Temp
                        
                        max_temp, c_x, c_y = ta.get_hot_spot_centroid(clean_temp_array, threshold=0.75)
                        display_data = image_Temp

                        # Update relative pixel coordinates
                        if max_temp is not None and c_x is not None and c_y is not None:
                            pixel_text.set_text(f"Relative (Pixels): X = {c_x:.1f} px, Y = {c_y:.1f} px  |  Peak: {max_temp:.2f} °C")
                            hot_marker.set_data([c_x], [c_y])
                        else:
                            pixel_text.set_text("Relative (Pixels): No hot spot detected above threshold")
                            hot_marker.set_data([], [])

                        # Update real-world gantry coordinates
                        if transform_matrix is not None and c_x is not None and c_y is not None:
                            hot_real_world = ta.get_mm_from_pixels(c_x, c_y, transform_matrix)
                            real_world_text.set_text(f"Real-World (Gantry): X = {hot_real_world[0]:.2f} mm,  Y = {hot_real_world[1]:.2f} mm")
                        else:
                            real_world_text.set_text("Real-World (Gantry): Calibration Matrix Unavailable")

                    # Draw / update image array in-place without rebuilding axes
                    if im_display is None:
                        im_display = ax.imshow(display_data, cmap='inferno', aspect='auto')
                        cbar = fig.colorbar(im_display, ax=ax, format='%.2f')
                        cbar.set_label('Temperature (°C)', rotation=270, labelpad=15)
                    else:
                        im_display.set_data(display_data)
                        im_display.set_clim(vmin=float(np.min(display_data)), vmax=float(np.max(display_data)))

                    # Process GUI events without raising the window or stealing focus
                    fig.canvas.draw_idle()
                    fig.canvas.flush_events()
                    time.sleep(0.001)

                    if keyboard.is_pressed('q'):
                        print('Program is closing...')
                        plt.close('all')
                        CONTINUE_RECORDING = False

                image_result.Release()

            except PySpin.SpinnakerException as ex:
                print('Error: %s' % ex)
                return False

        cam.EndAcquisition()

    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)
        return False

    return True


def run_single_camera(cam):
    """
    This function acts as the body of the example; please see NodeMapInfo example
    for more in-depth comments on setting up cameras.

    :param cam: Camera to run on.
    :type cam: CameraPtr
    :return: True if successful, False otherwise.
    :rtype: bool
    """
    try:
        result = True

        nodemap_tldevice = cam.GetTLDeviceNodeMap()

        # Initialize camera
        cam.Init()

        # Retrieve GenICam nodemap
        nodemap = cam.GetNodeMap()

        # Acquire images
        result &= acquire_and_display_images(cam, nodemap, nodemap_tldevice)

        # Deinitialize camera
        cam.DeInit()

    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)
        result = False

    return result


def main():
    """
    Example entry point; please see Enumeration example for more in-depth
    comments on preparing and cleaning up the system.

    :return: True if successful, False otherwise.
    :rtype: bool
    """
    result = True

    # Retrieve singleton reference to system object
    system = PySpin.System.GetInstance()

    # Get current library version
    version = system.GetLibraryVersion()
    print('Library version: %d.%d.%d.%d' % (version.major, version.minor, version.type, version.build))

    # Retrieve list of cameras from the system
    cam_list = system.GetCameras()

    num_cameras = cam_list.GetSize()

    print('Number of cameras detected: %d' % num_cameras)

    # Finish if there are no cameras
    if num_cameras == 0:

        # Clear camera list before releasing system
        cam_list.Clear()

        # Release system instance
        system.ReleaseInstance()

        print('Not enough cameras!')
        input('Done! Press Enter to exit...')
        return False

    # Run example on each camera
    for i, cam in enumerate(cam_list):

        print('Running example for camera %d...' % i)
        try:
                with Connection.open_serial_port("COM6") as connection:
                    connection.enable_alerts()
                    device_list = connection.detect_devices()
                    device = device_list[0]
                    y = device_list[1].get_axis(1)
                    x = device.get_axis(1)
                    z = device.get_axis(2)
                    
                    print("[MOTION] Homing Gantry...")
                    if not x.is_homed(): x.home(wait_until_idle=False)
                    if not y.is_homed(): y.home(wait_until_idle=False)
                    if not z.is_homed(): z.home(wait_until_idle=False)
                    x.wait_until_idle()
                    y.wait_until_idle()
                    z.wait_until_idle()
                    
                    # Start the background gantry thread
                    gantry_thread = threading.Thread(target=gantryControl, args=(x, y, z), daemon=True)
                    gantry_thread.start()

                    result &= run_single_camera(cam)

        except Exception as e:
            print(f"Error occurred while setting up gantry: {e}")
                
        print('Camera %d example complete... \n' % i)

    # Release reference to camera
    # NOTE: Unlike the C++ examples, we cannot rely on pointer objects being automatically
    # cleaned up when going out of scope.
    # The usage of del is preferred to assigning the variable to None.
    del cam

    # Clear camera list before releasing system
    cam_list.Clear()

    # Release system instance
    system.ReleaseInstance()

    return result

if __name__ == '__main__':
    main()

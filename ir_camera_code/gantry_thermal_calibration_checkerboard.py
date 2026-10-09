import sys
import PySpin
import cv2
import threading
import queue
import numpy as np
import thermal_analysis as ta

# Ensure we only keep the newest frame to prevent memory leaks and latency
frame_queue = queue.Queue(maxsize=1)
system_running = True

def camera_thread_function():
    global system_running
    system = PySpin.System.GetInstance()
    cam_list = system.GetCameras()
    
    if cam_list.GetSize() == 0:
        print('Not enough cameras!')
        cam_list.Clear()
        system.ReleaseInstance()
        system_running = False
        return

    cam = cam_list.GetByIndex(0)
    try:
        nodemap_tldevice = cam.GetTLDeviceNodeMap()
        cam.Init()
        nodemap = cam.GetNodeMap()

        # Force Radiometric Mode and NewestOnly buffer
        node_IRFormat = PySpin.CEnumerationPtr(nodemap.GetNode('IRFormat'))
        node_IRFormat.SetIntValue(node_IRFormat.GetEntryByName('Radiometric').GetValue())
        
        sNodemap = cam.GetTLStreamNodeMap()
        node_bufferhandling_mode = PySpin.CEnumerationPtr(sNodemap.GetNode('StreamBufferHandlingMode'))
        node_bufferhandling_mode.SetIntValue(node_bufferhandling_mode.GetEntryByName('NewestOnly').GetValue())

        # Set continuous acquisition
        node_acquisition_mode = PySpin.CEnumerationPtr(nodemap.GetNode('AcquisitionMode'))
        node_acquisition_mode.SetIntValue(node_acquisition_mode.GetEntryByName('Continuous').GetValue())

        # Retrieve Calibration details
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

        # Environmental constants
        Emiss, TRefl, TAtm, Humidity = 0.25, 293.15, 293.15, 0.55
        Dist, ExtOpticsTransmission = 2.5, 1
        TAtmC = TAtm - 273.15

        H2O = Humidity * np.exp(1.5587 + 0.06939 * TAtmC - 0.00027816 * TAtmC**2 + 0.00000068455 * TAtmC**3)
        Tau = X * np.exp(-np.sqrt(Dist) * (A1 + B1 * np.sqrt(H2O))) + (1 - X) * np.exp(-np.sqrt(Dist) * (A2 + B2 * np.sqrt(H2O)))
        r1 = ((1 - Emiss) / Emiss) * (R / (np.exp(B / TRefl) - F))
        r2 = ((1 - Tau) / (Emiss * Tau)) * (R / (np.exp(B / TAtm) - F))
        r3 = ((1 - ExtOpticsTransmission) / (Emiss * Tau * ExtOpticsTransmission)) * (R / (np.exp(B / TAtm) - F))
        K2 = r1 + r2 + r3

        cam.BeginAcquisition()
        print('Camera hardware initialized and streaming...')

        while system_running:
            image_result = cam.GetNextImage(1000)
            if not image_result.IsIncomplete():
                image_data = image_result.GetNDArray()
                
                # Apply Radiometric calculation
                image_Radiance = (image_data - J0) / J1
                image_Temp = (B / np.log(R / ((image_Radiance / Emiss / Tau) - K2) + F)) - 273.15
                
                # Push safely to main thread
                if not frame_queue.full():
                    frame_queue.put(image_Temp)
                    
            image_result.Release()
            
        cam.EndAcquisition()
        cam.DeInit()
        
    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)
    
    del cam
    cam_list.Clear()
    system.ReleaseInstance()


if __name__ == '__main__':
    print("Starting background camera thread...")
    cam_thread = threading.Thread(target=camera_thread_function)
    cam_thread.start()

    cv2.namedWindow("Thermal Calibration", cv2.WINDOW_NORMAL)
    print("\n--- CONTROLS ---")
    print("Press 'c' to run checkerboard calibration on the current frame.")
    print("Press 'q' to exit.\n")

    try:
        while system_running:
            if not frame_queue.empty():
                current_temp_array = frame_queue.get()
                
                # Normalize temp purely for visual display in the live feed
                vmin, vmax = np.percentile(current_temp_array, (2, 98))
                norm_img = np.clip((current_temp_array - vmin) * (255.0 / (vmax - vmin)), 0, 255).astype(np.uint8)
                display_img = cv2.applyColorMap(norm_img, cv2.COLORMAP_INFERNO)
                
                cv2.imshow("Thermal Calibration", display_img)
                
            key = cv2.waitKey(30) & 0xFF
            
            if key == ord('c') and 'current_temp_array' in locals():
                print("\nAttempting calibration...")
                # Pass the raw float array to your library function
                matrix, debug_img = ta.calibrate_with_checkerboard(current_temp_array, square_size_mm=30.0)
                
                # Show the resulting corner detections
                cv2.imshow("Calibration Debug", debug_img)
                cv2.waitKey(2000) # Pause on the diagnostic image for 2 seconds
                
                if matrix is not None:
                    print("Calibration Successful. Matrix saved.")
                    
            elif key == ord('q'):
                print("Exiting...")
                system_running = False
                
    except KeyboardInterrupt:
        system_running = False

    cam_thread.join()
    cv2.destroyAllWindows()
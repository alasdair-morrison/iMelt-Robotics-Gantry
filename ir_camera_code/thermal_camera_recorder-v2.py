import sys
import time
import queue
import threading
import numpy as np
import cv2
import PySpin

class AsyncVideoRecorder:
    """
    Asynchronous queue-backed video recorder using OpenCV VideoWriter.
    Prevents disk encoding stalls from dropping FLIR camera frames.
    """
    def __init__(self, filename="thermal_recording.mp4", fps=20.0, frame_size=(640, 480), max_queue_size=60):
        self.filename = filename
        self.fps = fps
        self.frame_size = frame_size
        self.queue = queue.Queue(maxsize=max_queue_size)
        self.stop_event = threading.Event()
        
        # Define MP4 container codec ('mp4v')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(self.filename, fourcc, self.fps, self.frame_size)
        
        if not self.writer.isOpened():
            raise RuntimeError(f"Failed to open VideoWriter for file: {self.filename}")
            
        # Spawn background writer thread
        self.worker_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self.worker_thread.start()

    def _writer_loop(self):
        """Consumer thread: pulls frames from queue and writes to disk."""
        while not self.stop_event.is_set() or not self.queue.empty():
            try:
                frame = self.queue.get(timeout=0.2)
                if frame is None:  # Shutdown sentinel
                    self.queue.task_done()
                    break
                self.writer.write(frame)
                self.queue.task_done()
            except queue.Empty:
                continue

        if self.writer is not None:
            self.writer.release()

    def add_frame(self, frame):
        """Non-blocking producer method. Drops frame if queue is full."""
        try:
            self.queue.put_nowait(frame)
        except queue.Full:
            pass  # Prioritize control/camera acquisition over video encoding

    def stop(self):
        """Gracefully flushes remaining frames and closes video file."""
        self.stop_event.set()
        try:
            self.queue.put_nowait(None)
        except queue.Full:
            pass
        if self.worker_thread.is_alive():
            self.worker_thread.join()


def record_thermal_stream(output_filename="thermal_recording.mp4", target_fps=20.0, 
                          auto_scale=True, vmin=20.0, vmax=200.0, colormap=cv2.COLORMAP_INFERNO):
    """
    Main capture loop: initializes PySpin system, acquires FLIR frames,
    converts radiometric data to color-mapped frames with automatic contrast AGC,
    and records asynchronously.
    
    Args:
        output_filename (str): Path to saved MP4 file.
        target_fps (float): Target video frame rate.
        auto_scale (bool): If True, uses dynamic 1st-99th percentile contrast stretching (like SpinView AGC)
                           so room-temperature scenes render crisp thermal details instead of a flat blur.
        vmin (float): Minimum temperature threshold (°C) if auto_scale is False.
        vmax (float): Maximum temperature threshold (°C) if auto_scale is False.
        colormap: OpenCV colormap (e.g., cv2.COLORMAP_INFERNO, cv2.COLORMAP_JET, cv2.COLORMAP_PLASMA).
    """
    system = PySpin.System.GetInstance()
    cam_list = system.GetCameras()
    
    num_cameras = cam_list.GetSize()
    if num_cameras == 0:
        cam_list.Clear()
        system.ReleaseInstance()
        print("[ERROR] No FLIR cameras detected on the system.")
        return

    cam = cam_list[0]
    recorder = None

    try:
        # Initialize camera
        cam.Init()
        
        # Configure acquisition to continuous streaming
        node_map = cam.GetNodeMap()
        node_acq_mode = PySpin.CEnumerationPtr(node_map.GetNode("AcquisitionMode"))
        if PySpin.IsAvailable(node_acq_mode) and PySpin.IsWritable(node_acq_mode):
            node_acq_mode_continuous = node_acq_mode.GetEntryByName("Continuous")
            node_acq_mode.SetIntValue(node_acq_mode_continuous.GetValue())

        # Begin acquisition
        cam.BeginAcquisition()
        print(f"[INFO] Streaming started. Recording to '{output_filename}'...")
        print(f"[INFO] Auto-scale (Dynamic AGC): {auto_scale} | Colormap: INFERNO")
        print("[INFO] Press Ctrl+C to stop recording and finalize video file.")

        frame_width = None
        frame_height = None

        while True:
            image_result = cam.GetNextImage(1000)
            
            if image_result.IsIncomplete():
                print(f"[WARN] Incomplete frame image status: {image_result.GetImageStatus()}")
                image_result.Release()
                continue

            # Extract raw image buffer as NumPy array
            raw_data = image_result.GetNDArray()
            
            if frame_width is None or frame_height is None:
                frame_height, frame_width = raw_data.shape[:2]
                # Initialize recorder once frame dimensions are known
                recorder = AsyncVideoRecorder(
                    filename=output_filename,
                    fps=target_fps,
                    frame_size=(frame_width, frame_height)
                )

            # Determine frame pixel values / temperature conversion
            # FLIR A-series in Linear10mK mode outputs counts where 1 count = 0.01 K
            # Standard raw Mono16 outputs 14/16-bit digital counts (0 - 65535)
            if raw_data.dtype == np.uint16:
                # Check if values are in Linear10mK range (e.g. 29315 = 293.15 K = 20 °C)
                sample_val = np.median(raw_data)
                if 20000 < sample_val < 45000:
                    # Linear10mK mode: count * 0.01 = Kelvin
                    temp_data = raw_data.astype(np.float32) * 0.01 - 273.15
                else:
                    # Generic Mono16 raw digital counts
                    temp_data = raw_data.astype(np.float32)
            else:
                temp_data = raw_data.astype(np.float32)

            # Contrast Normalization
            if auto_scale:
                # Dynamic AGC: 1st and 99th percentile contrast stretching (prevents flat maroon blurs)
                p_low = np.percentile(temp_data, 1)
                p_high = np.percentile(temp_data, 99)
                
                # Prevent division by zero for uniform scenes
                if p_high <= p_low:
                    p_high = p_low + 1.0
                    
                norm_frame = np.clip((temp_data - p_low) / (p_high - p_low) * 255.0, 0, 255).astype(np.uint8)
            else:
                # Fixed absolute temperature windowing (°C)
                norm_frame = np.clip((temp_data - vmin) / (vmax - vmin) * 255.0, 0, 255).astype(np.uint8)

            # Apply thermal colormap
            color_frame = cv2.applyColorMap(norm_frame, colormap)

            # Pass frame asynchronously to video recorder queue
            recorder.add_frame(color_frame)

            # Release PySpin image buffer
            image_result.Release()

    except KeyboardInterrupt:
        print("\n[INFO] Keyboard interrupt received. Stopping recording...")

    except Exception as e:
        print(f"[ERROR] An exception occurred during capture: {e}")

    finally:
        # Cleanup recorder and flush remaining frames
        if recorder is not None:
            print("[INFO] Flushing buffered frames and finalizing MP4 file...")
            recorder.stop()
            print(f"[SUCCESS] Video saved successfully to '{output_filename}'.")

        # De-initialize and release camera
        try:
            cam.EndAcquisition()
        except Exception:
            pass
            
        try:
            cam.DeInit()
        except Exception:
            pass

        del cam
        cam_list.Clear()
        system.ReleaseInstance()
        print("[INFO] PySpin camera instance released.")


if __name__ == "__main__":
    # By default, auto_scale=True applies dynamic AGC so room-temperature scenes show high contrast
    record_thermal_stream(output_filename="trial_run_thermal.mp4", target_fps=20.0, auto_scale=True)

import cv2
import numpy as np
import hailort  # Importing the hailort library

# Constants for the camera
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
FPS = 30

# Path to your .hef file
HEF_FILE_PATH = '/home/bode/Desktop/ev_monitoring_pi/yolov8l.hef'

def preprocess_frame(frame, input_shape):
    # Preprocess the camera frame before sending it to the Hailo-8 for inference
    resized_frame = cv2.resize(frame, (input_shape[1], input_shape[0]))
    return resized_frame

def main():
    # Open the USB camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS)

    if not cap.isOpened():
        print("Error: Could not open video stream from USB camera")
        return

    # Initialize the Hailo device
    with hailort.Device() as device:
        # Load the .hef model
        network_group = device.configure_network_group(HEF_FILE_PATH)
        
        # Get the input and output vstreams
        input_vstream = network_group.create_input_vstreams()[0]
        output_vstream = network_group.create_output_vstreams()[0]
        
        # Get the input shape (Height, Width) expected by the model
        input_shape = input_vstream.get_frame_shape()
        print(f"Model input shape: {input_shape}")

        while True:
            # Capture frame-by-frame from the USB camera
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to grab a frame from the camera")
                break
            
            # Preprocess the frame
            input_frame = preprocess_frame(frame, input_shape)

            # Add batch dimension and convert to numpy array for Hailo-8
            input_data = np.expand_dims(input_frame, axis=0)

            # Send data to Hailo-8 for inference
            input_vstream.write(input_data)
            output = output_vstream.read()

            # Process the output (this will depend on the model)
            print(f"Inference output: {output}")

            # Display the frame
            cv2.imshow('USB Camera Feed', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

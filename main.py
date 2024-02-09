import os
from ultralytics import YOLO
import cv2
import serial
import time
import matplotlib.pyplot as plt

#Access to the location of the trained model parameters  
model_path = os.path.join('.', 'runs', 'detect', 'train5_ant', 'weights', 'last.pt') #Need to have the model trained a priori, each parameter of this function is a folder

# ARDUINO
arduinoData = serial.Serial('COM6', 115200)
# Pause to allow the system set up
time.sleep(10)

# Load a model
model = YOLO(model_path)

# Open connected camera / make sure no other software is using the webcam
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_AUTOFOCUS, 0) #Activate webcam's auto focus in case this shows any improvements; 1=Activate, 0=Deactivate

# Initialize PD controller constants for each axis
base_kp_x = 7
base_kp_y = 7
base_kd_x = 0.7
base_kd_y = 0.7

# Initialize errors and previous errors
error_x = 0
error_y = 0
prev_error_x = 0
prev_error_y = 0
prev_time = time.time()

motor_x = 0
motor_y = 0

# Lists to store positions
errorx_plt = []
errory_plt = []
time_plt = []


ret, frame = cap.read()
if not ret:
    print("Failed to grab frame")
    exit()

H, W, _ = frame.shape
threshold = 0.3
desired_x = 0
desired_y = 0
max_motor_signal = 1400

#--------------------VIDEO---------------------------------------
# Define output and format of a video for further analysis
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
out_complete = cv2.VideoWriter('complete_video.mp4', fourcc, 25.0, (W,H)) #25.0 fps / also consider: 'int(cap.get(cv2.CAP_PROP_FPS))' instead
out_detection = cv2.VideoWriter('only_when_detected.mp4', fourcc, 25.0, (W,H)) 
#----------------------------------------------------------------


last_detection_time = time.time()
waiting_time = 0  # Initialize waiting_time
flagHome = False

start_time = time.time()

while ret:
    out_detection.write(frame) #Capture all frames in the output video
    
    current_time = time.time()
    dt = current_time - prev_time
    
    results = model(frame)[0]

    if results.boxes.data.numel() != 0:
        # Select the detection with the highest score
        best_detection = max(results.boxes.data, key=lambda x: x[4])
        x1, y1, x2, y2, score, class_id = best_detection

        if score >= threshold:
            out_detection.write(frame) #Capture the frames where an object is being detected 
            
            waiting_time = 0
            flagHome = False
            
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

            distance_x = int((x1 + x2) / 2)
            distance_y = int((y1 + y2) / 2)
            cv2.circle(frame, (distance_x, distance_y), 5, (0, 255, 0), -1)

            # Calculate the center of the bounding box
            bbox_center_x = distance_x - W // 2
            bbox_center_y = -(distance_y - H // 2)  # Inverting y-axis for Cartesian plane

            # Calculate the error as the difference from the center of the frame
            error_x = desired_x - bbox_center_x
            error_y = desired_y - bbox_center_y

            # Calculate the derivative terms (change in error / time)
            d_error_x = (error_x - prev_error_x) / dt if dt > 0 else 0
            d_error_y = (error_y - prev_error_y) / dt if dt > 0 else 0

            # Adjust gains based on error magnitude
            kp_x = base_kp_x + abs(error_x) * 0.03
            kp_y = base_kp_y + abs(error_y) * 0.03
            kd_x = base_kd_x
            kd_y = base_kd_y 
            
            # cv2.putText(frame, f"KP X: {kp_x} px", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            # cv2.putText(frame, f"KP Y: {kp_y} px", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.putText(frame, f"Pos X: {error_x} px", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Pos Y: {error_y} px", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Calculate the control signals
            motor_x = kp_x * error_x + kd_x * d_error_x
            motor_y = kp_y * error_y + kd_y * d_error_y

            # Limit the control signals
            motor_x = max(min(motor_x, max_motor_signal), -max_motor_signal)
            motor_y = max(min(motor_y, max_motor_signal), -max_motor_signal)

            # Send errors to Arduino for motor control
            error_str = f"{motor_x} {motor_y}\r"
            # arduinoData.write(error_str.encode())

            # Update previous errors and time
            prev_error_x = error_x
            prev_error_y = error_y
            prev_time = current_time

            # Store positions
            errorx_plt.append(error_x)
            errory_plt.append(error_y)
            time_plt.append(current_time - start_time)
            

    else:
        error_x = 0
        error_y = 0
        cv2.putText(frame, f"WARNING: No object detected.", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        arduinoData.write("0 0\r".encode())
        
        waiting_time += current_time - last_detection_time
        last_detection_time = current_time

        if (waiting_time >= 5) and (flagHome is False):
            # arduinoData.write("home\r".encode())
            flagHome = True
            motor_x2 = 30
            motor_y2 = 30
            
        elif (waiting_time >= 10) and (waiting_time <= 50):
            motor_x2 = motor_x2 + 1.5
            motor_y2 = motor_y2 + 1.5
            
            motor_x = round(motor_x2)
            motor_y = round(motor_y2)
            
            case = int(waiting_time) % 12  # This will cycle through 0, 1, 2, and 3 every second
            
            if case >= 0 and case <= 2:
                error_str = f"{-motor_x} {motor_y}\r"
            elif case >= 3 and case <= 5:
                error_str = f"{-motor_x} {-motor_y}\r"
            elif case >= 6 and case <= 8:
                error_str = f"{motor_x} {-motor_y}\r"
            elif case >= 9 and case <= 11:
                error_str = f"{motor_x} {motor_y}\r"

            arduinoData.write(error_str.encode())
            
        elif (waiting_time > 50):
            waiting_time = 0
            flagHome = False
            arduinoData.write("0 0\r".encode())
            
    # Display frame with predictions
    cv2.imshow('Live Predictions', frame)

    # If 'q' key is pressed, break the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Capture the next frame
    ret, frame = cap.read()

# Send zeros to stop motors
arduinoData.write("0 0\r".encode())

# Close Arduino communication
arduinoData.close()

# Plotting position ERROR
plt.plot(time_plt,errorx_plt)
plt.grid()
plt.title("X Error Over Time")
plt.ylabel("Error (x)")
plt.xlabel("Time (s)")
plt.show()

plt.plot(time_plt,errory_plt)
plt.grid()
plt.title("Y Error Over Time")
plt.ylabel("Error (y)")
plt.xlabel("Time")
plt.show()

cap.release()
out_complete.release()
out_detection.release()
cv2.destroyAllWindows()

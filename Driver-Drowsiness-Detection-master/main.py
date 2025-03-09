# Importing necessary libraries
import cv2
import numpy as np
import dlib
from imutils import face_utils
import pygame
from collections import deque
from twilio.rest import Client
import geocoder  # For getting the approximate location based on IP
import time

# Initialize pygame for sound alerts
pygame.mixer.init()

# Twilio setup
account_sid = "ACf15d07fe1731a268d########9a1"  # Replace with your Twilio Account SID
auth_token = "a7bc6b2c22e9########71b4c35190dafb"  # Replace with your Twilio Auth Token
twilio_phone_number = "+19########"  # Replace with your Twilio phone number
recipient_phone_number = "+91########"  # Replace with the phone number to receive SMS

# Initialize Twilio client
client = Client(account_sid, auth_token)

# Initialize the camera
cap = cv2.VideoCapture(0)

# Initialize face and landmark detectors
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Variables for tracking states
sleep = 0
drowsy = 0
active = 0
status = ""
color = (0, 0, 0)

# Timer and SMS setup
sleep_start_time = None
sms_sent = False

# Buffer to store EAR values for rolling average
ear_buffer = deque(maxlen=6)  # Store EAR of last 6 frames

# Define threshold values for states
EAR_SLEEP_THRESHOLD = 0.18  # Threshold for sleeping (eye completely closed)
EAR_DROWSY_THRESHOLD = 0.21  # Threshold for drowsy

# Threshold for detecting low-light conditions
BRIGHTNESS_THRESHOLD = 50  # Adjust this based on your environment

# Define sound file paths
sleeping_sound = "harsh.wav"
drowsy_sound = "normal.wav"


def play_sound(file):
    pygame.mixer.music.load(file)
    pygame.mixer.music.play()


def compute(ptA, ptB):
    dist = np.linalg.norm(ptA - ptB)
    return dist


def eye_aspect_ratio(eye):
    # Compute EAR using eye landmarks
    A = compute(eye[1], eye[5])
    B = compute(eye[2], eye[4])
    C = compute(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


def get_driver_location():
    g = geocoder.ip("me")  # Get location based on public IP
    return g.latlng  # Return latitude and longitude


def send_sms():
    global sms_sent
    if sms_sent:
        return

    try:
        # Get the driver's location
        location = get_driver_location()
        if location:
            lat, lng = location
            location_text = f"Driver's location: Latitude: {lat}, Longitude: {lng}"
        else:
            location_text = "Location unavailable."

        # Compose the SMS message
        message_body = f"The driver has been sleeping for more than 2 minutes. Please take action immediately.\n{location_text}"

        # Send SMS via Twilio
        message = client.messages.create(
            body=message_body, from_=twilio_phone_number, to=recipient_phone_number
        )
        print(f"SMS sent! Message SID: {message.sid}")
        sms_sent = True  # Mark that the SMS has been sent
    except Exception as e:
        print(f"Failed to send SMS: {e}")


def check_drowsiness(ear_avg):
    global sleep, drowsy, active, status, color, sleep_start_time, sms_sent
    if ear_avg < EAR_SLEEP_THRESHOLD:
        sleep += 1
        drowsy = 0
        active = 0
        if sleep_start_time is None:
            sleep_start_time = time.time()  # Start the timer when sleep is detected
        if sleep > 6:  # If eyes have been closed for long enough
            status = "SLEEPING !!!"
            color = (255, 0, 0)
            play_sound(sleeping_sound)
            if time.time() - sleep_start_time > 10:  # 2 minutes of sleeping
                send_sms()
    elif EAR_SLEEP_THRESHOLD <= ear_avg <= EAR_DROWSY_THRESHOLD:
        sleep = 0
        active = 0
        drowsy += 1
        sleep_start_time = None  # Reset the sleep timer
        sms_sent = False  # Reset SMS flag when driver is not asleep
        if drowsy > 6:
            status = "Drowsy !"
            color = (0, 0, 255)
            play_sound(drowsy_sound)
    else:
        drowsy = 0
        sleep = 0
        active += 1
        sleep_start_time = None  # Reset the sleep timer
        sms_sent = False  # Reset SMS flag when driver is active
        if active > 6:
            status = "Active :)"
            color = (0, 255, 0)


def apply_night_mode(frame):
    # Apply a filter to enhance visibility in low-light conditions
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.equalizeHist(frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    return frame


def is_low_light(frame):
    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Calculate the average brightness
    avg_brightness = np.mean(gray_frame)
    return avg_brightness < BRIGHTNESS_THRESHOLD


while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Create two copies of the frame for the different displays
    state_frame = frame.copy()  # This will show the status (Active, Drowsy, Sleeping)
    landmark_frame = frame.copy()  # This will show the landmarks

    # Check if the frame is in low light and apply night mode if necessary
    if is_low_light(frame):
        frame = apply_night_mode(frame)
        state_frame = apply_night_mode(state_frame)
        landmark_frame = apply_night_mode(landmark_frame)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)
        landmarks = face_utils.shape_to_np(landmarks)

        # Get the eye coordinates
        left_eye = landmarks[36:42]
        right_eye = landmarks[42:48]

        # Compute EAR for both eyes
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)

        # Average the EAR for both eyes
        ear = (left_ear + right_ear) / 2.0

        # Store the EAR in the buffer
        ear_buffer.append(ear)

        # Calculate the rolling average EAR
        ear_avg = np.mean(ear_buffer)

        # Check drowsiness using the rolling average EAR
        check_drowsiness(ear_avg)

        # Display the status on the "state" frame
        cv2.putText(
            state_frame,
            f"State: {status}",
            (100, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            color,
            3,
        )

        # Highlight the eyes on the "landmark" frame
        for x, y in landmarks[36:42]:  # Left eye landmarks
            cv2.circle(
                landmark_frame, (x, y), 2, (0, 255, 0), -1
            )  # Green for the left eye
        for x, y in landmarks[42:48]:  # Right eye landmarks
            cv2.circle(
                landmark_frame, (x, y), 2, (0, 255, 0), -1
            )  # Green for the right eye

        # Visualize other facial landmarks
        for x, y in landmarks:
            cv2.circle(landmark_frame, (x, y), 1, (255, 255, 255), -1)

    # Resize both frames to the same height for side-by-side display
    height = frame.shape[0]
    width = frame.shape[1]

    state_frame_resized = cv2.resize(state_frame, (width, height))
    landmark_frame_resized = cv2.resize(landmark_frame, (width, height))

    # Concatenate the two frames horizontally (side by side)
    combined_frame = np.hstack((state_frame_resized, landmark_frame_resized))

    # Show the combined frame
    cv2.imshow("State and Landmarks", combined_frame)

    key = cv2.waitKey(1)
    if key == 27:  # Exit if ESC is
        break

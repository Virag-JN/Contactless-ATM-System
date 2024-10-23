import cv2
from cvzone.HandTrackingModule import HandDetector
from time import sleep
import cvzone
from pynput.keyboard import Controller, Key
import os
import tensorflow as tf
import screeninfo
from screeninfo import get_monitors  # Ensure this is used correctly

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

# Initialize video capture and set resolution
cap = cv2.VideoCapture(0)


try:
    monitor = get_monitors()[0]  # Get the first monitor's dimensions
    screen_width = monitor.width
    screen_height = monitor.height
except Exception as e:
    print("Error retrieving screen dimensions:", e)
    screen_width, screen_height = 600, 900  # Fallback values


# Set the resolution
cap.set(3, screen_width)  # Width
cap.set(4, screen_height)  # Height

# Hand detector with updated detection confidence
detector = HandDetector(detectionCon=0.8, maxHands=1)

# Define keyboard layout with numbers 0-9 split into two rows
keys = [["1", "2", "3", "4"],
        ["5", "6", "7", "8"],
        ["9", "0", "Backspace"]]

# Initialize the virtual keyboard controller
keyboard = Controller()

# Button class with smaller button size
class Button:
    def __init__(self, pos, text, size=[100, 100]):  # Smaller buttons for better security
        self.pos = pos
        self.size = size
        self.text = text

# Create button list from keyboard layout
buttonList = []
for i in range(len(keys)):
    for j, key in enumerate(keys[i]):
        buttonList.append(Button([150 * j + 50, 150 * i + 100], key))  # Adjust positions to fit two rows

# Draw all buttons on the screen
def drawAll(img, buttonList):
    for button in buttonList:
        x, y = button.pos
        w, h = button.size
        # Draw rectangle without color flash
        cvzone.cornerRect(img, (x, y, w, h), l=20, t=3)
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), cv2.FILLED)
        cv2.putText(img, button.text, (x + 30, y + 70),  # Adjust text position for smaller buttons
                    cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)  # Reduce font size for clarity
    return img

# Initialize variable to track the last key pressed
last_key = None

# Initialize smoothing variables
previous_positions = []
num_frames_for_smoothing = 5  # Number of frames to average over
alpha = 0.7  # Low-pass filter smoothing factor (0 < alpha < 1)

# Main loop
while True:
    success, img = cap.read()  # Capture frame
    hands, img = detector.findHands(img)  # Detect hands and return landmarks
    
    img = drawAll(img, buttonList)  # Draw keyboard buttons

    if hands:
        lmList = hands[0]['lmList'] if 'lmList' in hands[0] else hands[0].lmList  # Adjust for correct format

        # Add the current index finger position to the list of previous positions
        previous_positions.append(lmList[8])

        # Keep only the last 'num_frames_for_smoothing' positions
        if len(previous_positions) > num_frames_for_smoothing:
            previous_positions.pop(0)

        # Calculate the average position of the index finger
        avg_x = sum([pos[0] for pos in previous_positions]) // len(previous_positions)
        avg_y = sum([pos[1] for pos in previous_positions]) // len(previous_positions)

        # Apply low-pass filter to smooth movements
        smooth_x = alpha * avg_x + (1 - alpha) * lmList[8][0]
        smooth_y = alpha * avg_y + (1 - alpha) * lmList[8][1]

        # Use the smoothed and averaged position to check for key presses
        for button in buttonList:
            x, y = button.pos
            w, h = button.size

            if x < smooth_x < x + w and y < smooth_y < y + h:
                cv2.rectangle(img, (x - 5, y - 5), (x + w + 5, y + h + 5), (175, 0, 175), cv2.FILLED)
                cv2.putText(img, button.text, (x + 30, y + 70),  # Adjust text position for smaller buttons
                            cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)

                # Measure distance between index and middle finger
                l, _ = detector.findDistance(lmList[8], lmList[12])  # No draw argument

                # Detect click if fingers are close enough
                if l < 20:  # Reduced sensitivity by increasing the threshold
                    if last_key != button.text:
                        if button.text == "Backspace":
                            keyboard.press(Key.backspace)  # Handle backspace as a special key
                        else:
                            keyboard.press(button.text)  # Handle normal keys
                        last_key = button.text  # Update last key pressed
                        sleep(0.3)  # Increase debounce time
                else:
                    last_key = None  # Reset if finger is not over a button

    # Show the image
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to close
        break

cap.release()
cv2.destroyAllWindows()

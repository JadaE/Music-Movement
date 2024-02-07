"""
First Goal: based on approx. of hand to camera + left/right, different changes to a noise (current output) will be made
Final Goal: with the combination of 2 cameras facing each other, it will be able to give an even more accurate sound adjustment
"""

import numpy as np
import cv2 as cv
import mediapipe as mp 
from google.protobuf.json_format import MessageToDict 
# import glob

mpHands = mp.solutions.hands 
hands = mpHands.Hands( 
    static_image_mode=False, 
    model_complexity=1,
    min_detection_confidence=0.75, 
    min_tracking_confidence=0.75, 
    max_num_hands=2) 
vid = cv.VideoCapture(0)

if not vid.isOpened():
    print("Camera cannot be opened")
    exit()

while True:
    #capture frame by frame
    ret, frame = vid.read()

    if not ret:
        print("Can't receive frame (stream end?)... exiting...")
        break

    # Flip the image's frame
    frame = cv.flip(frame, 1) 

    # PREV observations on the frame go here vvv
    # gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Convert BGR image to RGB image 
    imgRGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB) 

    # PREV Display the resulting frame
    # cv.imshow('frame', gray)

    # Process the RGB image 
    results = hands.process(imgRGB) 

    #PREV
    # if cv.waitKey(1) == ord('q'):
        # break
    
	# If hands are present in image's frame
    if results.multi_hand_landmarks: 

		# Both Hands are present in image(frame) 
        if len(results.multi_handedness) == 2: 
            # Display 'Both Hands' on the image 
            cv.putText(frame, 'Both Hands', (250, 50), 
						cv.FONT_HERSHEY_COMPLEX, 0.9, 
						(0, 255, 0), 2) 

		# If any hand present 
        else: 
            for i in results.multi_handedness: 
				
				# Return whether it is Right or Left Hand 
                label = MessageToDict(i)[ 
					'classification'][0]['label'] 

                if label == 'Left': 
					
					# Display 'Left Hand' on left side of window 
                    cv.putText(frame, label+' Hand', (20, 50), 
								cv.FONT_HERSHEY_COMPLEX, 0.9, 
								(0, 255, 0), 2) 

                if label == 'Right': 
					
					# Display 'Left Hand' on left side of window 
                    cv.putText(frame, label+' Hand', (460, 50), 
								cv.FONT_HERSHEY_COMPLEX, 
								0.9, (0, 255, 0), 2) 

	# Display Video and when 'q' is entered, destroy the window 
    cv.imshow('Image', frame) 
    if cv.waitKey(1) & 0xff == ord('q'): 
        break

# When everything done, release the capture
vid.release()
cv.destroyAllWindows()



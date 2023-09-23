import cv2 #OpenCV library
import mediapipe as mp #Pipe for Media Objects in video capture
import numpy as np   #For Canvas

#Variables
colorToUse = (255, 0, 0)                            #Color Using
mode = 0                                            #Drawing or chaning
imgCanvas = np.zeros((600, 1280, 3), np.uint8)      #Canvas to draw
xp, yp = 0, 0                                       #Previous cursor position

#define video capture object
cv = cv2.VideoCapture(0)

#Hand Objects
mpHands = mp.solutions.hands
hands = mpHands.Hands()

#Drawing Utils
mpDraw = mp.solutions.drawing_utils

#Infinite loop for continuous capture
while(1):
    
    #Capture the video frame by frame
    #Return and frame value
    ret, frame = cv.read()
    #flipping frame
    frame = cv2.flip(frame, 1)
    # resize image
    frame = cv2.resize(frame, (1280, 600), interpolation = cv2.INTER_AREA)
    #Converting BGR to RGB image for processing
    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #Getting RGB image results related to hand objects
    results = hands.process(imgRGB)
    
    #Creating Colours
    #Blue
    cv2.circle(frame,(40, 20), 15 , (255, 0, 0), cv2.FILLED)
    #Green
    cv2.circle(frame,(80, 20), 15 , (0, 255, 0), cv2.FILLED)
    #Red
    cv2.circle(frame,(120, 20), 15 , (0, 0, 255), cv2.FILLED)
    #Eraser
    cv2.circle(frame,(160, 20), 15 , (0, 0, 0), cv2.FILLED)

    #HighLight selected circle
    if colorToUse == (255, 0, 0):
        cv2.circle(frame,(40, 20), 15 , (255, 255, 255), 2)
    elif colorToUse == (0, 255, 0):
        cv2.circle(frame,(80, 20), 15 , (255, 255, 255), 2)
    elif colorToUse == (0, 0, 255):
        cv2.circle(frame,(120, 20), 15 , (255, 255, 255), 2)
    elif colorToUse == (0, 0, 0):
        cv2.circle(frame,(160, 20), 15 , (255, 255, 255), 2)
    
    #Whether hand is detected or not
    if results.multi_hand_landmarks:
        #Detecting Landmarks
        for handLms in results.multi_hand_landmarks:
            #Variable to store (x,y) of landmarks
            handLandmarks = []
            #mpDraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS)
            #Detecting landmarks and IDs of landmark
            for id, lm in enumerate(handLms.landmark):
                h, w, c = frame.shape
                #Getting Coordinates
                cx, cy = int(lm.x * w), int(lm.y * h)
                #appending all coordinates in a list
                handLandmarks.append([cx, cy])
                
                #Only If Index Finger
                if id==8:
                    if mode==1:
                        #Chaning Color
                        if cx>=25 and cx<=55 and cy>=5 and cy<=35:
                            colorToUse = (255, 0, 0)    #Blue
                        elif cx>=60 and cx<=95 and cy>=5 and cy<=35:
                            colorToUse = (0, 255, 0)    #Green
                        elif cx>=100 and cx<=135 and cy>=5 and cy<=35:
                            colorToUse = (0, 0, 255)    #Red
                        elif cx>=140 and cx<=175 and cy>=5 and cy<=35:
                            colorToUse = (0, 0, 0)      #Eraser
                        #Color marker on tip of index
                        cv2.rectangle(frame, (cx, cy), (cx+20, cy+20), colorToUse, cv2.FILLED)
                    else:
                        #Color marker on tip of index
                        cv2.circle(frame, (cx,cy), 5 , colorToUse, cv2.FILLED)
                    
            #Finger Count and Mode change
            #Index
            count = 0
            #Index Finger
            if handLandmarks[8][1] > handLandmarks[6][1]:
                count+=1
            #Middle Finger
            if handLandmarks[12][1] > handLandmarks[10][1]:
                count+=1
            #Drawing Mode
            if count==1:
                mode = 0
            else:
                mode = 1  #Color change Mode
                xp, yp = 0, 0
                
            #Drawing Mode
            if mode == 0:
                #Initialize
                if xp == 0 or yp == 0:
                    #To draw first point
                    xp, yp = handLandmarks[8][0], handLandmarks[8][1]
                #Increasing thickness of eraser
                if colorToUse != (0, 0, 0):
                    cv2.line(imgCanvas, (xp, yp), (handLandmarks[8][0], handLandmarks[8][1]), colorToUse, 10)
                else:
                    cv2.line(imgCanvas, (xp, yp), (handLandmarks[8][0], handLandmarks[8][1]), colorToUse, 30)
                #Updating previous position of points
                xp, yp = handLandmarks[8][0], handLandmarks[8][1]

    #Display Frame
    #Overlaying canvas to frame
    frame = cv2.addWeighted(frame, 0.6, imgCanvas, 1, 1)
    cv2.imshow("frame", frame)
    
    #'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
#After the loop releasing the capcutre object
cv.release()
#Destroying all windows
cv2.destroyAllWindows()
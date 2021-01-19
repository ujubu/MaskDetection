
# Importing the libraries
import cv2

# Loading the cascades
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')
mouth_cascade = cv2.CascadeClassifier('Mouth.xml')

# Adjust threshold value in range 80 to 105 based on your light.
bw_threshold = 80

# User message
font = cv2.FONT_HERSHEY_COMPLEX 	
org = (30, 30)
empty_font_color = (255, 0, 0)
weared_mask_font_color = (0, 255, 0)
not_weared_mask_font_color = (0, 0, 255)
thickness = 1
font_scale = 1
weared_mask = "Aferin maskeni takmissin"
not_weared_mask = "Maskeni Tak!"
not_weared_mask_smiled = "Ne guluyon, maskeni tak"

# Defining a function that will do the detections
def detect(gray, frame):
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    faces_bw = face_cascade.detectMultiScale(black_and_white, 1.1, 4)
    
    if(len(faces) == 0 and len(faces_bw) == 0):
        cv2.putText(frame, "Yuzun yok, nerede?", org, font, font_scale, empty_font_color, thickness, cv2.LINE_AA)
    elif(len(faces) == 0 and len(faces_bw) == 1):
        # It has been observed that for white mask covering mouth, with gray image face prediction is not happening
        cv2.putText(frame, weared_mask, org, font, font_scale, weared_mask_font_color, thickness, cv2.LINE_AA)
    else:
        # Draw rectangle on face
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]
            eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 22)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (125, 125, 0), 2)
            
            #Detect lips counters
            smiles = smile_cascade.detectMultiScale(roi_gray, 1.7, 22)
            mouth_rects = mouth_cascade.detectMultiScale(gray, 1.5, 5)

        # Face detected but Lips not detected which means person is wearing mask
        if(len(mouth_rects) == 0):
            cv2.putText(frame, weared_mask, org, font, font_scale, weared_mask_font_color, thickness, cv2.LINE_AA)
        else:
            if(len(smiles) == 1):
                for (sx, sy, sw, sh) in smiles:
                    cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (0, 0, 255), 2)
                cv2.putText(frame, not_weared_mask_smiled, org, font, font_scale, not_weared_mask_font_color, thickness, cv2.LINE_AA)
            else:
                for (mx, my, mw, mh) in mouth_rects:
                    if(y < my < y + h):
                        # Face and Lips are detected but lips coordinates are within face cordinates which `means lips prediction is true and
                        # person is not waring mask
                        cv2.putText(frame, not_weared_mask, org, font, font_scale, not_weared_mask_font_color, thickness, cv2.LINE_AA)
                        #cv2.rectangle(img, (mx, my), (mx + mh, my + mw), (0, 0, 255), 3)
                        break        
        
    return frame

# Doing some Face Recognition with the webcam
video_capture = cv2.VideoCapture(0)
while True:
    _, frame = video_capture.read()
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    (thresh, black_and_white) = cv2.threshold(gray, bw_threshold, 255, cv2.THRESH_BINARY)
    canvas = detect(gray, frame)
    cv2.imshow('Video', canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()

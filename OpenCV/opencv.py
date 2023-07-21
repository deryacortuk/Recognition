import cv2 

import imageio 

face_cascade = cv2.CascadeClassifier("haarcascade-frontalface-default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade-eye.xml")

def recognition(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y), (x+w, y+h), (255,0,0),2)
        gray_face = gray[y:y+h,x:x+w]
        org_face = frame[y:y+h,x:x+w]
        eyes = eye_cascade.detectMultiScale(gray_face, 1.1, 3)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(org_face, (ex,ey),(ex+ew, ey+eh),(0,255,0),2)
    return frame
image = imageio.imread('Foundation.jpg')
image = recognition(frame=image)
imageio.imwrite('output.jpg', image)

reader = imageio.get_reader("1.mp4")
fps = reader.get_meta_data()["fps"]
writer = imageio.get_writer("output.mp4",fps=fps)

for i, frame in enumerate(reader):
    frame = recognition(frame)
    writer.append_data(frame)
    print(i)
    
writer.close()
        
        
# Webcam recognition
  
# video_capture = cv2.VideoCapture(0)
# while True:
#     _, frame = video_capture.read()
#     canvas = recognition(frame)
#     cv2.imshow('Video', canvas)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# video_capture.release()
# cv2.destroyAllWindows()
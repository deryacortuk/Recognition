import cv2 


def human_detection(image):
    boxes, weights = hog.detectMultiScale(image,winStride=(8,8))
    person = 1
    for x,y,w,h in boxes:
        cv2.rectangle(image, (x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(image, f'person-{person}',(x,y-10),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,255,0),1)
        person += 1
    return image



# def human_detection_from_video(url):
#     cap = cv2.VideoCapture(0)
#     success, image = cap.read()
    
#     while success:
#         cv2.imshow("Human Detection from Video", human_detection(image))
        
 
    
        
    

if __name__ == "__main__":
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    
    img1 = cv2.imread("pedestrians-on-crosswalk.jpg")
    cv2.imshow("Human Detection", human_detection(img1))
    cv2.imwrite("new.jpg",human_detection(img1))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
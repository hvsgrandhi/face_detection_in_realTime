import cv2

#load pre trained data on face frontals from opencv (haarcascade is an algorithm)
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#used to capture video from the webcam 
webcam = cv2.VideoCapture(0)

#iterate forever over frames
while True:
    successful_frame_read, frame = webcam.read() #reads the current frame

    #each frame is converted into grayscaled image
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #detect faces and returns coordinates 
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

    # draw rectangles around the faces
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("The name of the window",frame)
    key = cv2.waitKey(1) #it going to wait for a key press to execute the code

    #stop the execution of code on the press of Q key(ascii code)
    if key==81 or key==113:
        break

webcam.release() #releases all the resources like webcam


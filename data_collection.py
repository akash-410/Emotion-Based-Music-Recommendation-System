import mediapipe as mp
import numpy as np
import cv2

cap = cv2.VideoCapture(0)  # capturing live video

name = input("Enter the name of the data : ") # saving one type of data

holistic = mp.solutions.holistic #
hands = mp.solutions.hands
holis = holistic.Holistic()
drawing = mp.solutions.drawing_utils

X = []  # main list to stor row
data_size = 0
while True:
    lst = []  # to store columns

    _, frm = cap.read()  # reading the frame

    frm = cv2.flip(frm, 1) # to flip the fram don't want mirror img

    res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)) # copy me likha h kyu kra
    #
    if res.face_landmarks:  # if some one in the frame
        for i in res.face_landmarks.landmark: #  we will iterate in the list face_landmarks.landmark: and will go to all this objects of data
            lst.append(i.x - res.face_landmarks.landmark[1].x)
            lst.append(i.y - res.face_landmarks.landmark[1].y)
    #         # why we are substracting postion in 2 d plan x,y
    #     you see that this particular face will have som elandmarks on x and y axix have some particular pos
    #      i am here and then i got here now i am at this pos so this will have going to diff pos points but this and this face both are ging to give
    #      same reaction but have diff values  we dont want this type of thing in our data - bcoz it will going to create unnecessary data size and data training
    # so we have solution for this
    #     we have a centre point data  of all of other key points with respect to nose point
    #      i.x -current x value and  and taking refernce point as res.face_landmarks.landmark[1].x

    # same done for right and left handmarks
        if res.left_hand_landmarks:
            for i in res.left_hand_landmarks.landmark:
                lst.append(i.x - res.left_hand_landmarks.landmark[8].x)
                lst.append(i.y - res.left_hand_landmarks.landmark[8].y)
        else: # if left hand is not in frame just store 0,0 in place of all those  42  why 42 bcoz this res.left_hand_landmarks.landmark has a size of 21 and 21 times 2 meaning for x and y gives 42 and we have 0,0
            for i in range(42):
                lst.append(0.0)
    #
        if res.right_hand_landmarks:
            for i in res.right_hand_landmarks.landmark:
                lst.append(i.x - res.right_hand_landmarks.landmark[8].x)
                lst.append(i.y - res.right_hand_landmarks.landmark[8].y)
        else:
            for i in range(42):
                lst.append(0.0)
    #
        X.append(lst)
        data_size = data_size + 1
    #
    drawing.draw_landmarks(frm, res.face_landmarks, holistic.FACEMESH_CONTOURS)
    drawing.draw_landmarks(frm, res.left_hand_landmarks, hands.HAND_CONNECTIONS)
    drawing.draw_landmarks(frm, res.right_hand_landmarks, hands.HAND_CONNECTIONS)
    #
    # cv2.putText(frm, str(data_size), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2) # how many data we collected size

    cv2.imshow("window", frm)

    if cv2.waitKey(1) == 27 or data_size > 99: # just taking 99 samples for each type of data in from of np array
    # if cv2.waitKey(1) == 27:    # # if user press scpace key destro all the windows
        cv2.destroyAllWindows()
        cap.release()     # realse the capture video
        break

np.save(f"{name}.npy", np.array(X))
print(np.array(X).shape) # just insure that we have taken some data at least it shoul not empty


# 1

#
#



# 2

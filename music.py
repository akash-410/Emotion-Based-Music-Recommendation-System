import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model
import webbrowser

model = load_model("model.h5")
label = np.load("labels.npy")
holistic = mp.solutions.holistic
hands = mp.solutions.hands
holis = holistic.Holistic()
drawing = mp.solutions.drawing_utils

st.header("Emotion Based Music Recommender")

if "run" not in st.session_state: # ye usi liye use kra jiska reasonniche imp me phle number pr diya h
    st.session_state["run"] = "true" # create dwith variable name run and initally i set true bcoz i want to run it

try:
    emotion = np.load("emotion.npy")[0] # ye 0 kyu liya bcoz ye arra format me h or hme vhi chaiye jo 0th elemnt pr  h in search query
except:
    emotion = ""

if not (emotion): # if emotion something i want to run webrtc
    st.session_state["run"] = "true"
else: # if there somethng in emotion i want to false
    st.session_state["run"] = "false"


class EmotionProcessor:
    def recv(self, frame): # to recieve all this frma in fucntion recv
        frm = frame.to_ndarray(format="bgr24") #format ky udiya bcoz open cv work with bgr format

        ##############################
        frm = cv2.flip(frm, 1)

        res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

        lst = []

        if res.face_landmarks:
            for i in res.face_landmarks.landmark:
                lst.append(i.x - res.face_landmarks.landmark[1].x)
                lst.append(i.y - res.face_landmarks.landmark[1].y)

            if res.left_hand_landmarks:
                for i in res.left_hand_landmarks.landmark:
                    lst.append(i.x - res.left_hand_landmarks.landmark[8].x)
                    lst.append(i.y - res.left_hand_landmarks.landmark[8].y)
            else:
                for i in range(42):
                    lst.append(0.0)

            if res.right_hand_landmarks:
                for i in res.right_hand_landmarks.landmark:
                    lst.append(i.x - res.right_hand_landmarks.landmark[8].x)
                    lst.append(i.y - res.right_hand_landmarks.landmark[8].y)
            else:
                for i in range(42):
                    lst.append(0.0)

            lst = np.array(lst).reshape(1, -1)

            pred = label[np.argmax(model.predict(lst))]

            print(pred)
            cv2.putText(frm, pred, (50, 50), cv2.FONT_ITALIC, 1, (255, 0, 0), 2)

            np.save("emotion.npy", np.array([pred])) # we are storing prediction in  np array  when we require dwe load it  to our file

        drawing.draw_landmarks(frm, res.face_landmarks, holistic.FACEMESH_TESSELATION,landmark_drawing_spec=drawing.DrawingSpec(color=(0, 0, 255), thickness=-1,circle_radius=1),connection_drawing_spec=drawing.DrawingSpec(thickness=1))
        drawing.draw_landmarks(frm, res.left_hand_landmarks, hands.HAND_CONNECTIONS)
        drawing.draw_landmarks(frm, res.right_hand_landmarks, hands.HAND_CONNECTIONS)

        ##############################

        return av.VideoFrame.from_ndarray(frm, format="bgr24") # we have to convert this frame to av format (google kr)
#  ek or issue use bhi solv ekrna h jab hmne  recommend me songs press kr diya camera fir bhi cpature kr rha h apn ko use bnd krna h to soltion ye rha
#  so we can use st.session state agr broser window open hui ki hme seeeston state ke ocdeko excute krna h

#  imp -note when we press button the prd we made it will goes to automatically erase bcoz vo loopke ander h bahr nikal ke save bhi krdo ke to bhi erase  ho jayegi to iska solution ye h ki
#  isliye hmne emotion file banyi agr vo empti h agr emotion="" chod dte to har bar file empty rhit jab bhi code run krte h hme ye nhi chaihiye use ham use krbana chahte as a search in utube

lang = st.text_input("Language")
singer = st.text_input("singer")

if lang and singer and st.session_state["run"] != "false": # lang and sing inpute denge tbi web cam open hoga or bad vala --
    webrtc_streamer(key="key", desired_playing_state=True,video_processor_factory=EmotionProcessor) # desired_playing_state=True-- strat button aa rha tha camera to direct  catch and video processor factory to procees all this frames in class which deined by user

btn = st.button("Recommend me songs")

if btn:
    if not (emotion): # agr face cptur ehi nhi hua to ye line ayegi
        st.warning("Please let me capture your emotion first")
        st.session_state["run"] = "true" # still true bcoz i dont open a tab
    else:
        webbrowser.open(f"https://www.youtube.com/results?search_query={lang}+{emotion}+song+{singer}") # a web broser tab ye normally search kra utube or fir link me nbs us name ki jagh variable dal die we import webbroser for it
        np.save("emotion.npy", np.array([""]))  # hr bar ek hi emotionse thodi kholenge reset kr do agli bar fir catch krenge
        st.session_state["run"] = "false" # after opening broser window i want tos et it flase











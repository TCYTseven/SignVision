import cv2
import openai
import os
import re
import requests
import time
from time import sleep
import mediapipe as mp
import random
from flask import Flask, render_template, Response, send_from_directory, request
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import numpy as np
import math

openai.api_key = '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'


class Card:
    def __init__(self, name):
        self.name = name
        self.repetitions = 0
        self.previous_ease_factor = 2.5
        self.previous_interval = 0

    def __repr__(self):
        return str(self.name)


lst  = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
lst.append('del')
lst.append('space')
lst.append('nothing')
cards = [Card(i) for i in lst]
graduated_cards = {}
iteration = 0



def calculate_interval(correct, time, card):
    global iteration
    iteration += 1
    if correct == True:
        if time <= 1:
            quality = 5
        elif time <= 3:
            quality = 4
        elif time <= 5:
            quality = 5
        elif time <= 7:
            quality = 2
        elif time <= 10:
            quality = 1
        else:
            quality = 0

        graduated_cards.append({card: iteration})
        cards.remove(card)
    else:
        quality = 0

    for var in graduated_cards:
        if var[card] - card.previous_interval <= 0:
            cards.append(var)
            graduated_cards.remove(var)

    if card.repetitions == 0:
        interval = 1
    elif card.repetitions == 1:
        interval = 6
    elif card.repetitions > 1:
        interval = card.previous_interval * card.previous_ease_factor

    ease_factor = card.previous_ease_factor + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))

    if ease_factor < 1.3:
        ease_factor = 1.3

    card.previous_ease_factor = ease_factor
    card.previous_interval = interval

    card.repetitions += 1

    return math.ceil(interval)


def select_card():
    return random.choice(cards)

data_generator = ImageDataGenerator(samplewise_center=True, samplewise_std_normalization=True)
MODEL_NAME = 'model.h5'
model = load_model(MODEL_NAME)
app = Flask(__name__)

# Initialize the camera globally
camera = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

classes_file = open("classes.txt")
classes_string = classes_file.readline()
classes = classes_string.split()
classes.sort()  # The predict function sends out output in sorted order.

@app.route('/')
def index():
    return render_template('index.html')

predicted_class = 'NA'
question = 'unset'


test_imgs =  []
def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break

        # Convert the frame to RGB format
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False

        # Process the frame with the hands object to detect hand landmarks
        results = hands.process(frame_rgb)

        # Convert the frame back to BGR format for display
        frame_rgb.flags.writeable = True
        frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Compute the bounding box around the hand
                x_min = int(min([lmk.x for lmk in hand_landmarks.landmark]) * frame.shape[1])
                x_max = int(max([lmk.x for lmk in hand_landmarks.landmark]) * frame.shape[1])
                y_min = int(min([lmk.y for lmk in hand_landmarks.landmark]) * frame.shape[0])
                y_max = int(max([lmk.y for lmk in hand_landmarks.landmark]) * frame.shape[0])
                
                # Adjust the bounding box coordinates for cropping
                x_min -= 30
                x_max += 60
                y_min -= 15
                y_max += 35

                # Ensure the coordinates are within the frame boundaries
                x_min = max(0, x_min)
                x_max = min(frame.shape[1], x_max)
                y_min = max(0, y_min)
                y_max = min(frame.shape[0], y_max)


                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                # Crop the frame around the bounding box
                cropped_image = frame[y_min:y_max, x_min:x_max]
                resized_frame = cv2.resize(cropped_image, (200, 200))
                reshaped_frame = (np.array(resized_frame)).reshape((1, 200, 200, 3))
                frame_for_model = data_generator.standardize(np.float64(reshaped_frame))
                test_imgs.append(frame_for_model)

                # Convert the cropped frame to JPEG format
                ret, buffer = cv2.imencode('.jpg', frame)
                res = buffer.tobytes()

                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + res + b'\r\n')
        else:
            # If no hand landmarks are detected, yield the original frame
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        # Delay to achieve the desired FPS
        cv2.waitKey(4)





@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture')
def capture():
    folder_name = 'snapshots'
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        
    snapshot = request.args.get('snapshot')
    file_name = 'snapshot{}.png'.format(snapshot)
    snapshot_path = os.path.join(folder_name, file_name)

    success, frame = camera.read()

    if success:
        cv2.imwrite(snapshot_path, frame)
        return "Snapshot captured: {}".format(file_name)
    else:
        return "Failed to capture snapshot."

@app.route('/download_frame')
def download_frame():
    folder_name = 'downloadLogs'
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        
    random_number = random.randint(1, 10000)
    file_name = 'current_frame_{}.png'.format(random_number)
    frame_path = os.path.join(folder_name, file_name)

    success, frame = camera.read()

    if success:
        cv2.imwrite(frame_path, frame)
        return send_from_directory(folder_name, file_name, as_attachment=False)
    else:
        return "Failed to capture frame."



@app.route('/campreview')
def camp_review():
    return render_template('campreview.html')

@app.route('/longtermlearn')
def longtermlearn():
    
    return render_template('longtermlearn.html')

@app.route('/quiz')
def quiz():
    return render_template('quiz.html')

@app.route('/learntest')
def learntest():
    global question
    question = select_card()
    return render_template('learntest.html', predict_result=predicted_class, randomgenned=question)

@app.route('/process')
def process():
    global question
    max_mean = -1000000000000000
    final_pred  = None
    sample =  []
    pred_classes = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 0, 'F': 0, 'G': 0, 'H': 0, 'I': 0, 'J': 0,
              'K': 0, 'L': 0, 'M': 0, 'N': 0, 'O': 0, 'P': 0, 'Q': 0, 'R': 0, 'S': 0, 'T': 0,
              'U': 0, 'V': 0, 'W': 0, 'X': 0, 'Y': 0, 'Z': 0, 'space': 0, 'nothing': 0, 'del': 0}
    
    pred_probs = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 0, 'F': 0, 'G': 0, 'H': 0, 'I': 0, 'J': 0,
              'K': 0, 'L': 0, 'M': 0, 'N': 0, 'O': 0, 'P': 0, 'Q': 0, 'R': 0, 'S': 0, 'T': 0,
              'U': 0, 'V': 0, 'W': 0, 'X': 0, 'Y': 0, 'Z': 0, 'space': 0, 'nothing': 0, 'del': 0}
    for i in range(10):
        sample.append(random.choice(test_imgs))
    for img in sample:
        prediction = model.predict(img)
        predicted_class,presiction_probability = classes[np.argmax(prediction)], prediction[0, prediction.argmax()]
        pred_classes[predicted_class] += 1
        pred_probs[predicted_class] += presiction_probability
    
    print(pred_classes)
    for var in pred_classes:
        if pred_classes[var] == 0:
            mean = -1000000000000000
        else:
            mean = pred_classes[var]
        if mean > max_mean:
            max_mean = mean
            final_pred  = var
    #if final_pred == True:
    #    pass


    calculate_interval(final_pred==question, 5, question)
    
    return render_template('process.html', predictedresult=final_pred, correctanswer= question) #pass in the predict result

# def is_valid_input(input_text):
#     # Remove any non-alphanumeric characters from the input
#     cleaned_input = re.sub(r'\W+', '', input_text)

#     # Check if the cleaned input is at least 5 characters long
#     if len(cleaned_input) < 5:
#         return False
#     else:
#         return True

def is_valid_input(input_text):
    # Remove any non-alphanumeric characters from the input
    cleaned_input = re.sub(r'\W+', '', input_text)

    # Check if the cleaned input is at least 5 characters long
    if len(cleaned_input) < 5:
        return False
    else:
        return True

def bot(prompt,
        engine='text-davinci-003',
        temp=0.9,
        top_p=1.0,
        tokens=1000,
        freq_pen=0.0,
        pres_pen=0.5,
        stop=['<<END>>']):
  max_retry = 1
  retry = 0
  while True:
    try:
      response = openai.Completion.create(engine=engine,
                                          prompt=prompt,
                                          temperature=temp,
                                          max_tokens=tokens,
                                          top_p=top_p,
                                          frequency_penalty=freq_pen,
                                          presence_penalty=pres_pen,
                                          stop=[" User:", " AI:"])
      text = response['choices'][0]['text'].strip()
      print(text)
      return text
    except Exception as oops:
      retry += 1
      if retry >= max_retry:
        return "GPT3 error: %s" % oops
      print('Error communicating with OpenAI:', oops)
      sleep(1)


@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    botresponse = bot(prompt=userText)
    return str(botresponse)





if __name__ == '__main__':
    app.run(debug=True)

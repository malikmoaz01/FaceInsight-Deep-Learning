import cv2
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import time
from transformers import AutoTokenizer,AutoModelForCausalLM
import torch

nltk.download('punkt')   
nltk.download('stopwords')


def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return None, None
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(image, scaleFactor=1.1 , minNeighbors=5)

    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        face = image[y:y+h, x:x+w]
        face = cv2.resize(face, (224, 224))
        face = face / 255.0
        return face, faces[0]
    
    elif len(faces) > 1:
        raise Exception("Multiple faces detected in the image. Please provide an image with only one face.")
    
    else:
        print("No face detected in the image.")
        return None, None



def predict_gender_age(image_path):

    gender_model_file = "dataset/gender_deploy.prototxt"
    gender_weights_file = "dataset/gender_net.caffemodel"
    age_model_file = "dataset/age_deploy.prototxt"
    age_weights_file = "dataset/age_net.caffemodel"

    gender_net = cv2.dnn.readNetFromCaffe(gender_model_file, gender_weights_file)
    age_net = cv2.dnn.readNetFromCaffe(age_model_file, age_weights_file)
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return None, None

    blob = cv2.dnn.blobFromImage(image, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
    
    gender_net.setInput(blob)
    gender_detections = gender_net.forward()
    gender_list = ['Male', 'Female']
    gender = gender_list[gender_detections[0].argmax()]

    age_net.setInput(blob)
    age_detections = age_net.forward()
    age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
    age_index = age_detections[0].argmax()
    age = age_list[age_index]

    return gender, age

def detect_spots_yolo(image_path):

    net = cv2.dnn.readNet("dataset/yolov3.weights", "dataset/yolov3.cfg")

    with open("dataset/coco.names", "r") as f:
        classes = f.read().strip().split("\n")
    
    layer_names = net.getLayerNames()
    output_layers = ['yolo_82', 'yolo_94']

    image = cv2.imread(image_path)
    height, width, _ = image.shape

    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)

    outs = net.forward(output_layers)

    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id == 0: 
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    spots_detected = len(indexes) > 0
    return spots_detected

def detect_skin_issues(image_path, face_coordinates, age):
    spots_detected = detect_spots_yolo(image_path)
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return {}

    (x, y, w, h) = face_coordinates
    face = image[y:y+h, x:x+w]
    gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    acne_mask = cv2.inRange(face, (0, 0, 120), (50, 50, 255))
    pigmentation_mask = cv2.inRange(face, (30, 30, 30), (150, 150, 150))
    dark_circles_mask = cv2.inRange(gray_face, 30, 100)
    pimples_mask = cv2.inRange(face, (0, 0, 120), (50, 50, 255))
    redness_mask = cv2.inRange(face, (0, 0, 100), (100, 100, 255))
    
    wrinkles_mask = cv2.Canny(gray_face, 100, 200)
    
    cv2.imwrite("debug_acne_mask.jpg", acne_mask)
    cv2.imwrite("debug_pigmentation_mask.jpg", pigmentation_mask)
    cv2.imwrite("debug_dark_circles_mask.jpg", dark_circles_mask)
    cv2.imwrite("debug_redness_mask.jpg", redness_mask)
    cv2.imwrite("debug_pimples_mask.jpg", pimples_mask)
    cv2.imwrite("debug_wrinkles_mask.jpg", wrinkles_mask)

    acne_detected = cv2.countNonZero(acne_mask) > 50
    spots_detected = spots_detected 
    pimples_detected = cv2.countNonZero(pimples_mask) > 50
    pigmentation_detected = cv2.countNonZero(pigmentation_mask) > 100
    dark_circles_detected = cv2.countNonZero(dark_circles_mask) > 100
    redness_threshold = 100
    redness_detected = np.mean(redness_mask) > redness_threshold
    wrinkles_detected = cv2.countNonZero(wrinkles_mask) > 1000

    skin_issues = {
        "Acne": acne_detected,
        "Spots": spots_detected,
        "Pimples": pimples_detected,
        "Pigmentation": pigmentation_detected,
        "Dark Circles": dark_circles_detected,
        "Redness": redness_detected,
        "Wrinkles": wrinkles_detected
    }

    return skin_issues

def get_user_response(prompt):
    response = input(prompt)
    return response

def analyze_response(response):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(response)
    filtered_tokens = [word.lower() for word in tokens if word.lower() not in stop_words and word not in string.punctuation]
    return any(word in filtered_tokens for word in ["yes", "yeah", "yep", "sure", "y"])



def print_with_delay(text, delay=0.009):
    for char in text:
        print(char, end='', flush=True)
        time.sleep(delay)
    print()

def generate_suggestions(issue, gender, age):
    suggestions = ""
    if issue == "Acne":
        suggestions += "Suggestions for Acne:\n"
        suggestions += "- Keep your face clean. \n"
        suggestions += "- Use oil-free skin products. \n"
        suggestions += "- Consider over-the-counter treatments with benzoyl peroxide or salicylic acid. \n"
        suggestions += "- Maintain a healthy diet with plenty of fruits and vegetables. \n"
        suggestions += "- Drink plenty of water to stay hydrated. \n"
        suggestions += "- Avoid touching your face unnecessarily. \n"
        suggestions += "- Wash your pillowcases and bedsheets regularly. \n"
        suggestions += "- Use non-comedogenic makeup products. \n"
        suggestions += "- Consider visiting a dermatologist for personalized treatment options. \n"
        suggestions += "- Manage stress through relaxation techniques or hobbies.\n\n"
    elif issue == "Spots":
        suggestions += "Suggestions for Spots:\n"
        suggestions += "- Avoid touching or picking at spots. \n"
        suggestions += "- Keep your skin moisturized. \n"
        suggestions += "- Consider using products with ingredients like tea tree oil or witch hazel. \n"
        suggestions += "- Use sunscreen to prevent spots from darkening in the sun. \n"
        suggestions += "- Eat a balanced diet rich in vitamins and antioxidants. \n"
        suggestions += "- Get regular exercise to improve circulation and skin health. \n"
        suggestions += "- Practice stress-reducing activities such as yoga or meditation. \n"
        suggestions += "- Get enough sleep to allow your skin to regenerate. \n"
        suggestions += "- Try over-the-counter spot treatments with benzoyl peroxide or salicylic acid. \n"
        suggestions += "- Consult with a dermatologist for professional advice and treatment options.\n\n"
    elif issue == "Pimples":
        suggestions += "Suggestions for Pimples:\n"
        suggestions += "- Wash your face twice daily with a gentle cleanser. \n"
        suggestions += "- Avoid using harsh scrubs or exfoliants. \n"
        suggestions += "- Use non-comedogenic moisturizers and makeup products. \n"
        suggestions += "- Apply spot treatments containing benzoyl peroxide or salicylic acid. \n"
        suggestions += "- Avoid picking or squeezing pimples, as this can lead to scarring. \n"
        suggestions += "- Eat a balanced diet with plenty of fruits, vegetables, and whole grains. \n"
        suggestions += "- Keep hair clean and away from the face to prevent oil transfer. \n"
        suggestions += "- Use oil-free sunscreen to protect your skin from sun damage. \n"
        suggestions += "- Manage stress through relaxation techniques or hobbies. \n"
        suggestions += "- Consider prescription medications or treatments from a dermatologist if over-the-counter options are not effective.\n\n"
    elif issue == "Pigmentation":
        suggestions += "Suggestions for Pigmentation:\n"
        suggestions += "- Use sunscreen with a high SPF to prevent further darkening of pigmented areas. \n"
        suggestions += "- Consider topical treatments containing ingredients like hydroquinone or kojic acid. \n"
        suggestions += "- Use gentle exfoliants to help fade pigmentation over time. \n"
        suggestions += "- Consult with a dermatologist for options like chemical peels or laser therapy. \n"
        suggestions += "- Eat a diet rich in antioxidants to promote skin health. \n"
        suggestions += "- Avoid prolonged sun exposure, especially during peak hours. \n"
        suggestions += "- Keep skin moisturized to prevent dryness, which can accentuate pigmentation. \n"
        suggestions += "- Use makeup or concealer to cover pigmented areas if desired. \n"
        suggestions += "- Be patient, as it may take time for pigmentation to fade with treatment. \n"
        suggestions += "- Consider wearing protective clothing, such as hats and long sleeves, when outdoors.\n\n"
    elif issue == "Dark Circles":
        suggestions += "Suggestions for Dark Circles:\n"
        suggestions += "- Get enough sleep each night (7-9 hours for adults). \n"
        suggestions += "- Elevate your head while sleeping to reduce fluid retention under the eyes. \n"
        suggestions += "- Apply cold compresses or chilled cucumber slices to reduce puffiness. \n"
        suggestions += "- Use an eye cream containing ingredients like vitamin C or caffeine to brighten the under-eye area. \n"
        suggestions += "- Avoid rubbing or pulling on the delicate skin around the eyes. \n"
        suggestions += "- Stay hydrated by drinking plenty of water throughout the day. \n"
        suggestions += "- Use sunscreen to protect the skin from sun damage, which can worsen dark circles. \n"
        suggestions += "- Manage allergies or nasal congestion, which can contribute to dark circles. \n"
        suggestions += "- Consider cosmetic treatments like fillers or laser therapy for persistent dark circles. \n"
        suggestions += "- Conceal dark circles with makeup products specifically designed for the under-eye area.\n\n"
    elif issue == "Redness":
        suggestions += "Suggestions for Redness:\n"
        suggestions += "- Use a gentle cleanser and avoid harsh scrubbing, which can irritate the skin. \n"
        suggestions += "- Apply a soothing moisturizer with ingredients like aloe vera or chamomile. \n"
        suggestions += "- Use products labeled for sensitive skin and free of common irritants like fragrance and alcohol. \n"
        suggestions += "- Avoid hot showers or baths, as hot water can exacerbate redness. \n"
        suggestions += "- Protect your skin from extreme temperatures and wind with appropriate clothing or a scarf. \n"
        suggestions += "- Consider topical treatments like azelaic acid or niacinamide to reduce redness. \n"
        suggestions += "- Use a green-tinted primer or color-correcting makeup to neutralize redness. \n"
        suggestions += "- Manage stress through relaxation techniques like deep breathing or meditation. \n"
        suggestions += "- Avoid triggers like spicy foods, alcohol, and caffeine, which can worsen redness. \n"
        suggestions += "- Consult with a dermatologist for prescription medications or laser treatments for persistent redness.\n\n"
    elif issue == "Wrinkles":
        suggestions += "Suggestions for Wrinkles:\n"
        suggestions += "- Use a moisturizer with ingredients like retinol or hyaluronic acid to hydrate and plump the skin.\n "
        suggestions += "- Protect your skin from the sun by wearing sunscreen daily.\n "
        suggestions += "- Avoid smoking and limit alcohol consumption, as these can accelerate skin aging.\n"
        suggestions += "- Eat a diet rich in antioxidants and omega-3 fatty acids to support skin health.\n"
        suggestions += "- Stay hydrated by drinking plenty of water throughout the day.\n"
        suggestions += "- Get regular exercise to improve circulation and maintain overall health.\n"
        suggestions += "- Use skincare products containing peptides or growth factors to stimulate collagen production.\n"
        suggestions += "- Consider cosmetic treatments like Botox or dermal fillers for more significant wrinkle reduction.\n"
        suggestions += "- Practice facial exercises to tone and strengthen facial muscles. \n"
        suggestions += "- Get enough sleep each night to allow your skin to repair and regenerate.\n\n"
    suggestions += f"\nPredicted Gender: {gender}\n"
    suggestions += f"Predicted Age Range: {age}\n"

    return suggestions

def chat_with_gpt2():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    conversation_history = "" 
    while True:
        user_input = input("You: ")
        conversation_history += f"User: {user_input}\n"

        input_text = conversation_history[-1024:]  
        input_ids = tokenizer.encode(input_text, return_tensors="pt")

        with torch.no_grad():
            output = model.generate(input_ids, max_length=200, pad_token_id=tokenizer.eos_token_id)
        bot_reply = tokenizer.decode(output[0], skip_special_tokens=True)
        print("Bot:", bot_reply)
        conversation_history += f"Bot: {bot_reply}\n"


def main(image_path):
    face, face_coordinates = preprocess_image(image_path)
    if face is not None:
        gender, age = predict_gender_age(image_path)
        if gender and age:
            print(f"Gender: {gender}")
            print(f"Age Range: {age}")
            skin_issues = detect_skin_issues(image_path, face_coordinates, age)
            print("Skin Issues Detected:")
            total_suggestions = 0
            for issue, detected in skin_issues.items():
                print(f"{issue}: {'Yes' if detected else 'No'}")

                if detected:
                    response = get_user_response(f"Would you like suggestions for {issue.lower()}? (yes/no): ")
                    if analyze_response(response):
                        print_with_delay(f"Providing suggestions for {issue}: ...")

                        suggestions = generate_suggestions(issue, gender, age)
                        print_with_delay(suggestions)
                        total_suggestions += 1

                        if total_suggestions >= 10:
                            break

            if total_suggestions == 0:
                print_with_delay("No suggestions available for the detected skin issues.")

            response = get_user_response("Are you satisfied with the suggestions? If not, you can chat with our chatbot for more assistance. Would you like to chat? (yes/no): ")
            if analyze_response(response):
                print_with_delay("Sure! Let me connect you to our chatbot...")
                chat_with_gpt2() 

        else:
            print("Failed to predict gender or age.")
    else:
        print("No face detected in the image.")

if __name__ == '__main__':
    image_path = '1.jpg'
    main(image_path)
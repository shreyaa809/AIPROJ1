import random
import json
import pickle
import numpy as np
import tensorflow as tf

import nltk
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('omw-1.4')

from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import load_model

lemmatizer = WordNetLemmatizer()
with open('intents.json', encoding='utf-8') as file:
    intents = json.load(file)

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbotmodel.h5')


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)


def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': float(r[1])})
    return return_list

dean_map = {

    'academics': ("Dr. Anthony Xavior M", "dean.acad@vit.ac.in"),
    'academic research': ("Dr. Rajasekaran C", "dean.ar@vit.ac.in"),
    'sas': ("Dr. Karthikeyan K", "dean.sas@vit.ac.in"),
    'school of advanced sciences': ("Dr. Karthikeyan K", "dean.sas@vit.ac.in"),
    'sbst': ("Dr. Suneetha V", "dean.sbst@vit.ac.in"),
    'school of bio sciences': ("Dr. Suneetha V", "dean.sbst@vit.ac.in"),
    'sce': ("Dr. Saravana Kumar M P", "dean.sce@vit.ac.in"),
    'school of civil engineering': ("Dr. Saravana Kumar M P", "dean.sce@vit.ac.in"),
    'select': ("Dr. Kowsalya M", "dean.select@vit.ac.in"),
    'electrical engineering': ("Dr. Kowsalya M", "dean.select@vit.ac.in"),
    'sense': ("Dr. Jasmin Pemeena Priyadarisini M", "dean.sense@vit.ac.in"),
    'electronics engineering': ("Dr. Jasmin Pemeena Priyadarisini M", "dean.sense@vit.ac.in"),
    'shine': ("Dr. Geetha Manivasagam", "dean.shine@vit.ac.in"),
    'shine (healthcare)': ("Dr. Geetha Manivasagam", "dean.shine@vit.ac.in"),
    'smec': ("Dr. Kuppan P", "dean.smec@vit.ac.in"),
    'mechanical engineering': ("Dr. Kuppan P", "dean.smec@vit.ac.in"),
    'ssl': ("Dr. Selvam V", "dean.ssl@vit.ac.in"),
    'social sciences': ("Dr. Selvam V", "dean.ssl@vit.ac.in"),
    'vaial': ("Dr. Rajendran R", "dean.vaial@vit.ac.in"),
    'v-sign': ("Dr. Arun Tom Mathew", "dean.vsign@vit.ac.in"),
    'vit bs': ("Dr. Mary Cherian", "dean.vitbs@vit.ac.in"),
}


def find_dean_info(user_input):
    
    low = user_input.lower()
    
    for key in dean_map:
        if key in low:
            name, email = dean_map[key]
            return f"Dean: {name} — Email: {email}"
    
    tokens = [t.strip('.,') for t in low.split()]
    for t in tokens:
        if t in dean_map:
            name, email = dean_map[t]
            return f"Dean: {name} — Email: {email}"
    return None


def get_response(intents_list, intents_json, user_message=None):
    """Returns a response string. If the predicted intent is 'dean_info', it will try to dynamically
    provide dean information using the dean_map and the raw user message."""
    if not intents_list:
        return "Sorry, I didn't understand that. Can you rephrase?"

    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']

    if tag == 'dean_info' and user_message:
        dean_answer = find_dean_info(user_message)
        if dean_answer:
            return dean_answer
        for i in list_of_intents:
            if i['tag'] == tag:
                return random.choice(i['responses'])

    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    else:
        result = "Sorry, I couldn't find a suitable response."

    return result


print("VitAsk is running! (type 'quit' to exit)")

while True:
    pattern_to_tag = {}
    for it in intents['intents']:
        tag = it['tag']
        for p in it.get('patterns', []):
            pattern_to_tag[p.lower().strip()] = tag
    
    def direct_pattern_match(user_message):
        msg = user_message.lower().strip()
        if msg in pattern_to_tag:
            return pattern_to_tag[msg]
        for pat, tag in pattern_to_tag.items():
            if pat and pat in msg:
                return tag
        return None
    message = input("")
    if message.lower() in ('quit', 'exit'):
        print("Goodbye!")
        break
    forced_tag = direct_pattern_match(message)
    if forced_tag:
    # return a static response for the forced tag
        for it in intents['intents']:
            if it['tag'] == forced_tag:
               res = random.choice(it['responses'])
               break
    else:
        ints = predict_class(message)
        res = get_response(ints, intents, user_message=message)
    print(res)



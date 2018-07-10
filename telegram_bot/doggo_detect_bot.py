import json
import time

import requests
import numpy as np
import cv2
from urllib.parse import quote_plus
from keras.models import load_model

from image_classifier_model import predict_breed


with open('telegram_bot_token.json') as file:
    TOKEN = json.load(file)['token']
URL = 'https://api.telegram.org/bot{}/' .format(TOKEN)


def get_url(url):
    response = requests.get(url)
    content = response.content.decode("utf8")
    return content


def get_json_from_url(url):
    content = get_url(url)
    js = json.loads(content)
    return js


def get_updates(offset=None):
    url = URL + "getUpdates?timeout=100"
    if offset:
        url += "&offset={}".format(offset)
    js = get_json_from_url(url)
    return js


def send_message(text, chat_id, message_id):
    text = quote_plus(text)
    url = URL + "sendMessage?text={}&chat_id={}&reply_to_message_id={}".format(text, chat_id, message_id)
    get_url(url)


def get_last_update_id(updates):
    update_ids = []
    for update in updates["result"]:
        update_ids.append(int(update["update_id"]))
    return max(update_ids)


def echo_all(updates, model):
    for update in updates["result"]:
        message_id = update['message']['message_id']
        chat = update["message"]["chat"]["id"]
        sender_name = update['message']['from']['first_name']
        outbound_text = "Sup {}, you didn't send me an image leh".format(sender_name)
        if update['message']['chat']['type'] != 'private':
            if 'text' in update['message']:
                inbound_text = update['message']['text']
            elif 'caption' in update['message']:
                inbound_text = update['message']['caption']
            else:
                continue
            if '@DoggoDetectBot' not in inbound_text:
                continue
        try:
            if 'photo' in update['message']:
                file_id = update['message']['photo'][0]['file_id']
            elif update['message']['document']['mime_type'] == 'image/jpeg':
                file_id = update['message']['document']['file_id']
            else:
                continue
            image = download_image(file_id)
            breed_info = predict_breed(model, image)
            breed = breed_info[0]
            breed_prob = round(breed_info[1] * 100, 2)
            outbound_text = "Hola {}, I am {}% sure that this is a {}, what do you think?" \
                .format(sender_name, breed_prob, breed)
        except Exception as e:
            print(e)
        send_message(outbound_text, chat, message_id)


def download_image(file_id):
    filepath_info_url = URL + 'getFile?file_id={}' .format(file_id)
    filepath_info_json = get_json_from_url(filepath_info_url)
    filepath = filepath_info_json['result']['file_path']
    image_url = 'https://api.telegram.org/file/bot{}/{}' .format(TOKEN, filepath)
    image_data = requests.get(image_url).content
    arr = np.asarray(bytearray(image_data), dtype=np.uint8)
    image = cv2.imdecode(arr, -1)
    return image


def main(model_name='doggo_classifier_model.h5'):
    # load image classifier model
    doggo_model = load_model('../' + model_name)
    last_update_id = None
    while True:
        print(time.ctime())
        updates = get_updates(last_update_id)
        if len(updates["result"]) > 0:
            last_update_id = get_last_update_id(updates) + 1
            echo_all(updates, doggo_model)
        time.sleep(0.5)


if __name__ == '__main__':
    main()

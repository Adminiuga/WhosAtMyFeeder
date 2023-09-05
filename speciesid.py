import json
import logging
import multiprocessing
import sqlite3
import sys
import time
from datetime import datetime
from io import BytesIO

import numpy as np
import paho.mqtt.client as mqtt
import requests
import yaml
from PIL import Image, ImageOps
from tflite_support.task import core, processor, vision

from queries import get_common_name
from webui import app

classifier = None
config = None
firstmessage = True

DBPATH = "./data/speciesid.db"
DEFAULT_MQTT_PORT = 1833
DEFAULT_INSECURE_TLS = False
LOGGER_FMT = (
    "%(asctime)s.%(msecs)03d %(levelname)s (%(threadName)s) [%(name)s] %(message)s"
)
LOGGER_DATE_FMT = "%Y-%m-%d %H:%M:%S"
LOGGER_DEFAULT_LEVEL = logging.INFO

logging.basicConfig(format=LOGGER_FMT, datefmt=LOGGER_DATE_FMT)
_LOGGER = logging.getLogger(__name__)


def classify(image):
    tensor_image = vision.TensorImage.create_from_array(image)

    categories = classifier.classify(tensor_image)

    return categories.classifications[0].categories


def on_connect(client, userdata, flags, rc):
    _LOGGER.debug("MQTT Connected")

    # we are going subscribe to frigate/events and look for bird detections there
    client.subscribe(config["frigate"]["main_topic"] + "/events")


def on_disconnect(client, userdata, rc):
    if rc != 0:
        _LOGGER.warning("Unexpected mqtt disconnection: %s, trying to reconnect", rc)
        while True:
            try:
                client.reconnect()
                break
            except Exception as e:
                _LOGGER.error(
                    "Reconnection failed due to %s, retrying in 60 seconds", e
                )
                time.sleep(60)
    else:
        _LOGGER.info("MQTT Expected disconnection: %s", rc)


def set_sublabel(frigate_url, frigate_event, sublabel):
    post_url = frigate_url + "/api/events/" + frigate_event + "/sub_label"

    # frigate limits sublabels to 20 characters currently
    if len(sublabel) > 20:
        sublabel = sublabel[:20]

        # Create the JSON payload
    payload = {"subLabel": sublabel}

    # Set the headers for the request
    headers = {"Content-Type": "application/json"}

    # Submit the POST request with the JSON payload
    response = requests.post(post_url, data=json.dumps(payload), headers=headers)

    # Check for a successful response
    if response.status_code == 200:
        _LOGGER.debug("Sublabel set successfully to: %s", sublabel)
    else:
        _LOGGER.warning("Failed to set sublabel. Status code: %s", response.status_code)


def on_message(client, userdata, message):
    conn = sqlite3.connect(DBPATH)

    global firstmessage
    if not firstmessage:
        # Convert the MQTT payload to a Python dictionary
        payload_dict = json.loads(message.payload)

        # Extract the 'after' element data and store it in a dictionary
        after_data = payload_dict.get("after", {})
        _LOGGER.debug("mqtt event: %s", after_data)

        if (
            after_data["camera"] in config["frigate"]["camera"]
            and after_data["label"] == "bird"
        ):
            frigate_event = after_data["id"]
            frigate_url = config["frigate"]["frigate_url"]
            snapshot_url = (
                frigate_url + "/api/events/" + frigate_event + "/snapshot.jpg"
            )

            _LOGGER.info(
                "Getting image for event: %s from %s", frigate_event, snapshot_url
            )
            # Send a GET request to the snapshot_url
            params = {"crop": 1, "quality": 95}
            response = requests.get(snapshot_url, params=params)

            _LOGGER.debug("Getting snapshot '%s' status_code: %d", response.status_code)

            # Check if the request was successful (HTTP status code 200)
            if response.status_code == 200:
                # Open the image from the response content and convert it to a NumPy array
                image = Image.open(BytesIO(response.content))

                file_path = "fullsized.jpg"  # Change this to your desired file path
                image.save(
                    file_path, format="JPEG"
                )  # You can change the format if needed

                # Resize the image while maintaining its aspect ratio
                max_size = (224, 224)
                image.thumbnail(max_size)

                # Pad the image to fill the remaining space
                padded_image = ImageOps.expand(
                    image,
                    border=(
                        (max_size[0] - image.size[0]) // 2,
                        (max_size[1] - image.size[1]) // 2,
                    ),
                    fill="black",
                )  # Change the fill color if necessary

                file_path = "shrunk.jpg"  # Change this to your desired file path
                padded_image.save(
                    file_path, format="JPEG"
                )  # You can change the format if needed

                np_arr = np.array(padded_image)

                categories = classify(np_arr)
                category = categories[0]
                index = category.index
                score = category.score
                display_name = category.display_name
                category_name = category.category_name

                _LOGGER.info(
                    "Snapshot index: %s, score: %s, category: %s",
                    index,
                    score,
                    category,
                )

                if (
                    index != 964 and score > config["classification"]["threshold"]
                ):  # 964 is "background"
                    cursor = conn.cursor()

                    # Check if a record with the given frigate_event exists
                    cursor.execute(
                        "SELECT * FROM detections WHERE frigate_event = ?",
                        (frigate_event,),
                    )
                    result = cursor.fetchone()

                    if result is None:
                        # Insert a new record if it doesn't exist
                        _LOGGER.debug(
                            "No record yet for '%s' event. Storing.", frigate_event
                        )
                        cursor.execute(
                            """  
                            INSERT INTO detections (detection_time, detection_index, score,  
                            display_name, category_name, frigate_event, camera_name) VALUES (?, ?, ?, ?, ?, ?, ?)  
                            """,
                            (
                                formatted_start_time,
                                index,
                                score,
                                display_name,
                                category_name,
                                frigate_event,
                                after_data["camera"],
                            ),
                        )
                        # set the sublabel
                        set_sublabel(
                            frigate_url, frigate_event, get_common_name(display_name)
                        )
                    else:
                        _LOGGER.debug(
                            "There is already a record for '%s' event. Checking score",
                            frigate_event,
                        )
                        # Update the existing record if the new score is higher
                        existing_score = result[3]
                        if score > existing_score:
                            _LOGGER.debug(
                                "New score for '%s' event is higher. Updating record with higher score.",
                                frigate_event,
                            )
                            cursor.execute(
                                """  
                                UPDATE detections  
                                SET detection_time = ?, detection_index = ?, score = ?, display_name = ?, category_name = ?  
                                WHERE frigate_event = ?  
                                """,
                                (
                                    formatted_start_time,
                                    index,
                                    score,
                                    display_name,
                                    category_name,
                                    frigate_event,
                                ),
                            )
                            # set the sublabel
                            set_sublabel(
                                frigate_url,
                                frigate_event,
                                get_common_name(display_name),
                            )
                        else:
                            _LOGGER.debug(
                                "New score for '%s' event is lower.", frigate_event
                            )

                    # Commit the changes
                    conn.commit()

            else:
                _LOGGER.error(
                    "Error: Could not retrieve the image. Status code: %s",
                    response.status_code,
                )

    else:
        firstmessage = False
        _LOGGER.debug("skipping first message")

    conn.close()


def setupdb():
    _LOGGER.debug("Setting up '%s' database", DBPATH)
    conn = sqlite3.connect(DBPATH)
    cursor = conn.cursor()
    cursor.execute(
        """    
        CREATE TABLE IF NOT EXISTS detections (    
            id INTEGER PRIMARY KEY AUTOINCREMENT,  
            detection_time TIMESTAMP NOT NULL,  
            detection_index INTEGER NOT NULL,  
            score REAL NOT NULL,  
            display_name TEXT NOT NULL,  
            category_name TEXT NOT NULL,  
            frigate_event TEXT NOT NULL UNIQUE,
            camera_name TEXT NOT NULL 
        )    
    """
    )
    conn.commit()

    conn.close()


def load_config():
    global config
    file_path = "./config/config.yml"
    with open(file_path, "r") as config_file:
        config = yaml.safe_load(config_file)


def run_webui():
    host = config["webui"]["host"]
    port = config["webui"]["port"]
    _LOGGER.info("Starting flask app for '%s' host and %s port", flush=True)
    app.run(debug=False, host=host, port=port)


def run_mqtt_client():
    mqtt_host = config["frigate"]["mqtt_server"]
    mqtt_port = config["frigate"].get("mqtt_port", DEFAULT_MQTT_PORT)

    _LOGGER.info("Starting MQTT client. Connecting to: %s:%s", mqtt_host, mqtt_port)

    now = datetime.now()
    current_time = now.strftime("%Y%m%d%H%M%S")
    client = mqtt.Client("birdspeciesid" + current_time)
    client.on_message = on_message
    client.on_disconnect = on_disconnect
    client.on_connect = on_connect
    # check if we are using authentication and set username/password if so
    if config["frigate"]["mqtt_auth"]:
        username = config["frigate"]["mqtt_username"]
        password = config["frigate"]["mqtt_password"]
        client.username_pw_set(username, password)

    if config["frigate"].get("mqtt_use_tls", False):
        ca_certs = config["frigate"].get("mqtt_tls_ca_certs")
        client.tls_set(ca_certs)
        client.tls_insecure_set(
            config["frigate"].get("mqtt_tls_insecure", DEFAULT_INSECURE_TLS)
        )

    client.connect(config["frigate"]["mqtt_server"], mqtt_port)
    client.loop_forever()


def main():
    _LOGGER.info("Python version: %s, Version info: %s", sys.version, sys.version_info)

    load_config()
    log_level = config.get("logging", {}).get("default")
    if log_level is not None:
        try:
            logging.getLogger("").setLevel(log_level.upper())
        except ValueError:
            _LOGGER.error(
                "Unknown '%s' default logging level. Check the configuration file",
                log_level,
            )

    # Initialize the image classification model
    model = config["classification"]["model"]
    _LOGGER.debug("Initializing '%s' model", model)

    base_options = core.BaseOptions(file_name=model, use_coral=False, num_threads=4)

    # Enable Coral by this setting
    classification_options = processor.ClassificationOptions(
        max_results=1, score_threshold=0
    )
    options = vision.ImageClassifierOptions(
        base_options=base_options, classification_options=classification_options
    )

    # create classifier
    global classifier
    classifier = vision.ImageClassifier.create_from_options(options)

    # setup database
    setupdb()
    _LOGGER.info("Starting threads for Flask and MQTT")
    flask_process = multiprocessing.Process(target=run_webui)
    mqtt_process = multiprocessing.Process(target=run_mqtt_client)

    flask_process.start()
    mqtt_process.start()

    flask_process.join()
    mqtt_process.join()


if __name__ == "__main__":
    _LOGGER.debug("Calling Main")
    main()

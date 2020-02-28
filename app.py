import time
import json
import base64
from io import BytesIO
import numpy as np

from PIL import Image

from flask import Flask, request, send_file, g
from flask_cors import CORS

from backend import YOLOv3
import local_config

app = Flask(__name__)
CORS(app)

model = YOLOv3()

@app.route('/')
def index():
  """
  returns the list of available routes
  """
  return repr(['/inference', '/object-detection', " Go to /test GET method to run detection"])

@app.route('/inference')
def inference():
  return send_file('object_detection.html')

@app.route('/object-detection', methods=['POST'])
def object_detection_api():
  try:
#    encoded_img = base64.b64encode((request.files['image']).read())
     out_boxes, out_scores, class_ids, class_names = model.object_detect(request.json['imgstring'])
  except KeyError:
    # image field not provided
    print("what is happening!!!!!!!!!!!!!!!!!")
    return 'bad request', 400
    # expose server error

  result = {
    'time'  : time.time(),
    'out_boxes': out_boxes.tolist(),
    'out_scores' : out_scores.tolist(),
    'class_ids' : class_ids.tolist(),
    'class_names' : class_names
  }
  json_string = json.dumps(result)
  return json_string

@app.before_request
def set_start_time():
  g.request_start_time = time.time()


@app.after_request
def log_response_time(res):
  delta_t = time.time() - g.request_start_time
  print(f'Request completed in {delta_t}s')
  return res


if __name__ == "__main__":
  app.run(host='0.0.0.0', port=local_config.PORT)

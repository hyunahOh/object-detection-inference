This is project for making API in task 1 : object detection for 10 objects that are commonly seen on roads.

This project has similar API format as the project named 'task2-api'.

model: YOLO v3 (https://github.com/qqwweee/keras-yolo3). <br></br>
Initially when you git clone the source code, follow instructions in the README. </br>
That is, you have to follow these steps: </br>
1. get weights:  wget https://pjreddie.com/media/files/yolov3.weights 
2. get model:  python convert.py yolov3.cfg yolov3.weights model_data/yolo.h5
3. runn:  python yolo_video.py --image (for image detection) 

In coco_annotation.py, I used train2014.json instead of 2017.

To run my codes,
python app_object_detection.py



As of Aug 28th, I added app_json.py and backend_json.py.
I originally wanted to show everything: image, bounding box information, and elapsed time.
But my code only shows image... 

# Smart-Social-Distancing-Detector using TensorFlow Lite (TfLite)

Application written in C++ to detect the social distancing between humans using SSD MobileNet with TensorFlow Lite Inference Engine engine as backend.
The applications shows a video stream with the results of social distancing inferences amongs the humans present in the input images.

## Algorithm:
1. Use SSD Mobilenet pre-trained model to detect Humans. If not re-trained, the filter all other objects except Humans.
2. Compute the centroids of the bounding Boxes
3. Compute Euclidean Distance between the pairs of Humans as detected
4. Based on thresholds, classify the distancing into 3 categories.
    1. No adequate Social Distancing
    2. Adequate Social Distancing
    3. Safe Social Distancing
   
 Note: The calculation of Euclidean Distance and infernce of social distancing is written assuming the camera is in same plane of the scene of detection, example, webcam of a Laptop, smartphone camera. If this needs to be used with a CCTV at a corner, then the inference of Euclidean distances needs to be modified.
 
 ## Dependencies
 OpenCV
 
 TfLite
 
 ## Building
 1. Build the tensorflowlite static library by using the make script in tensorflow/lite/tools/make/build_lib.sh (for x86 systems)
 2. export SDK_DIR=tensorflow
 3.	g++ smart_social_distancing.cpp -o smart_social_distancing -I${SDK_DIR}/ \
	-I${SDK_DIR}/tensorflow/lite/tools/make/downloads/absl/ \
	-I${SDK_DIR}/tensorflow/lite/tools/make/downloads/flatbuffers/include/ \
	-L${SDK_DIR}/lite/tools/make/gen -ltensorflow-lite -lrt -lpthread `pkg-config --cflags --libs opencv4`
4. Any building or linking errors can only arise from above include paths. Just make them synchronized with the Tf installation

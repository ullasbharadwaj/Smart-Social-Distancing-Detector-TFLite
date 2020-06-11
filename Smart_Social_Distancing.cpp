/*****************************************************************************
      Smart Social Distancing Detection Application
      based on TensorFlow Lite Inference Engine

      Model Used for Detection: SSD MobileNets

      Author: Ullas Bharadwaj
****************************************************************************/

/* Tensorflow related Includes */
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/string_util.h"
#include "tensorflow/lite/model.h"

/* OpenCV related Includes */
#include <opencv2/opencv.hpp>

/* Generic Includes */
#include <fstream>
#include <sstream>
#include <iostream>
#include <stdint.h>
#include <sys/ioctl.h>
#include <fcntl.h>
#include <cmath>
#include <chrono>
#include <time.h>
/* Required for Frame Buffer operations */
#include <linux/fb.h>

using namespace cv;
using namespace dnn;
using namespace std;

float nms_threshold;
int No_Detections_Allowed;
float Prob_Threshold;
vector<string> classes;
int cam_width, cam_height;

std::string keys =
    "{ help  h     | | Print help message. }"
    "{ device d      |  0 | camera device number. }"
    "{ input i       | traffic.mp4 | Path to input resized_image or video file. Skip this argument to capture frames from a camera Default: traffic.mp4. }"
    "{ model m        | | Path to the TFLite Model. }"
    "{ classes c     | | Path to a text file with names of classes to label detected objects. }"
    "{ thr t        | .4 | Confidence threshold. }";

/********************************
Function to calculate Euclidean
Norm between the centroids of the
two neighboring Humans in detection
results
********************************/
float Calc_Euclidean_Distance(Rect& rectA, Rect& rectB) {
  float distanceA = (2 * 3.14 * 180) / (rectA.width + rectA.height * 360) * 1000 + 3;
  float distanceB = (2 * 3.14 * 180) / (rectB.width + rectB.height * 360) * 1000 + 3;
  float centerAx = (rectA.x + (rectA.width/2))/ distanceA;
  float centerAy = (rectA.y - rectA.height/2) / distanceA;
  float centerBx = (rectB.x + (rectB.width/2)) / distanceB;
  float centerBy = (rectB.y - rectB.height/2) / distanceB;
  float Euclidean = sqrt( ((centerAx - centerBx)*(centerAx - centerBx)) + ((centerAy - centerBy)*(centerAy - centerBy)) ) ;
  float limit = 30 * min(distanceA, distanceB) / max(distanceA, distanceB);
  // cout << "Euclidean : " << Euclidean << " Limit: " << limit << endl;
  if (Euclidean > limit) {
    if (Euclidean < limit + limit/2)
      return 1;   /* Moderate Social Distancing */
    else
      return 2;   /* Adequate Social Distancing */
  }
  return 0;       /* No Social Distancing */

}

/********************************
Function to draw the bounding boxes
around the detected objects
********************************/
void draw_bounding_boxes(int classId, float conf, int left, int top, int right, int bottom, Mat& image, float color)
{
  /* Draw a rectangle around the detected Object */
  Mat overlay;
  double alpha = 0.3;

  image.copyTo(overlay);

  if (color == 0)     // No Social Distance
    rectangle(overlay, Point(left, top), Point(right, bottom), Scalar(0, 0, 255), FILLED);
  else if (color == 1)   // Moderate Social Distance
    rectangle(overlay, Point(left, top), Point(right, bottom), Scalar(0, 165, 255),FILLED);
  else if (color == 2)   // Adequate Social Distance
    rectangle(overlay, Point(left, top), Point(right, bottom), Scalar(0, 255, 0),FILLED);

    addWeighted(overlay, alpha, image, 1 - alpha, 0, image);

  rectangle(image, Point(left, top), Point(right, bottom), Scalar(0, 255, 0), 4);

  string label = format("%.0f", conf*100) + "%";
  if (!classes.empty())
  {
      CV_Assert(classId < (int)classes.size());
      label = classes[classId+1] + ": " + label;
  }

  int baseLine;
  Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

  top = max(top, labelSize.height);
  rectangle(image, Point(left, top - labelSize.height),
            Point(left + labelSize.width, top + baseLine), Scalar::all(255), FILLED);
  putText(image, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.5, Scalar());
}

void post_processing(Mat& input_image, unique_ptr<tflite::Interpreter>& interpreter) {

      /* TFLite Detection_Post Process return 4 ouputs,
          Detection Boxes: Coordinates of the Bounding boxes,
          Detection Classes: Class IDs corresponding to detection each box,
          Detection Scores: Confidence value corresponding to each box
          Number of Detected Boxes: Number of detected objects        */
          /* Define output tensors */

      TfLiteTensor* Detection_Boxes = nullptr;
      TfLiteTensor* Detection_Classes = nullptr;
      TfLiteTensor* Number_Boxes = nullptr;
      TfLiteTensor* Detection_Scores = nullptr;

      Detection_Boxes             = interpreter->tensor(interpreter->outputs()[0]);
      auto detection_boxes        = Detection_Boxes->data.f;

      Detection_Classes           = interpreter->tensor(interpreter->outputs()[1]);
      auto detection_classes      = Detection_Classes->data.f;

      Detection_Scores            = interpreter->tensor(interpreter->outputs()[2]);
      auto detection_scores       = Detection_Scores->data.f;

      Number_Boxes                = interpreter->tensor(interpreter->outputs()[3]);
      auto nums_boxes             = Number_Boxes->data.f;

      /* Number of Detections allowed is decided at the time of generating
      tflite_graph.pb using export_ssd_graph.py. Refer to prepare_models.sh in
      the repository at src/dependenceies/ */
      vector<float> locations;
      vector<float> cls;

      for (int i = 0; i < No_Detections_Allowed /* No of Detections */; i++){
        auto output = detection_boxes[i];
        locations.push_back(output);
        cls.push_back(detection_classes[i]);
      }

      int count=0;
      int ymin,xmin, ymax, xmax;
      vector<Rect> boxes;
      vector<float> scores;
      vector<int> classIDs;

      for(int j = 0; j < *nums_boxes; j+=4){

      /* Ignore boxes corresponding to objects other than human */
        if(cls[count] != 0)
          continue;

        ymin = locations[j+0] * cam_height;
        xmin = locations[j+1] * cam_width;
        ymax = locations[j+2] * cam_height;
        xmax = locations[j+3] * cam_width;

        auto width = xmax - xmin;
        auto height = ymax - ymin;

        float score = detection_scores[count];

        boxes.push_back(Rect(xmin, ymin, width, height));
        scores.push_back(score);
        classIDs.push_back(cls[count]);

        count+=1;
      }
      vector<int> indices;

      /* Perform Non-Max Suppression to remove redundant bounding boxes */
      NMSBoxes(boxes, scores, Prob_Threshold, nms_threshold, indices);

      /* Draw bounding boxes around the detected Objects */
      for (size_t i = 0; i < indices.size(); ++i)
      {
          float social_distance_level = 2;
          int idx = indices[i];
          Rect box = boxes[idx];

          for (size_t j = 0; j < indices.size() ; ++j) {
            int index = indices[j];
            if (i != j) {
              social_distance_level = Calc_Euclidean_Distance(box, boxes[index]);
              if (social_distance_level == 0) {
                break;
              }
            }
          }

          draw_bounding_boxes(classIDs[idx], scores[idx], box.x, box.y,
                   box.x + box.width, box.y + box.height, input_image, social_distance_level);
      }
}

int main(int argc, char** argv) {

    nms_threshold = 0.3;
    No_Detections_Allowed = 20;
    double previous_fps = 0;


    CommandLineParser parser(argc, argv, keys);
    parser = CommandLineParser(argc, argv, keys);
    parser.about("Use this application to run object detection based on deep learning networks using TensorFlowLite.");

    if (argc == 1 || parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }

    Prob_Threshold = parser.get<float>("thr");
    string modelPath = parser.get<String>("model");
    string labelPath = parser.get<String>("classes");


    static const std::string Window_Name = "Smart Social Distancing Detector in TesnorflowLite";
    static const std::string Trackbar_Name = "Smart Social Distancing Detector in TesnorflowLite";
    namedWindow(Window_Name, WINDOW_NORMAL);
    resizeWindow(Window_Name, 600,600);
    int initialConf = (int)(Prob_Threshold * 100);
    createTrackbar(Trackbar_Name, Window_Name, &initialConf, 99);

		// Load model
		unique_ptr<tflite::FlatBufferModel> model =
		tflite::FlatBufferModel::BuildFromFile(modelPath.c_str());
		// Build the interpreter
		tflite::ops::builtin::BuiltinOpResolver resolver;
		unique_ptr<tflite::Interpreter> interpreter;
		tflite::InterpreterBuilder(*model.get(), resolver)(&interpreter);

		// Resize input tensors, if desired.

		TfLiteTensor* Detection_Boxes = nullptr;
		TfLiteTensor* Detection_Classes = nullptr;
		TfLiteTensor* Number_Boxes = nullptr;
    TfLiteTensor* Detection_Scores = nullptr;

		VideoCapture cam;
    /* Open a mp4 video file or a Camera stream */
    if (parser.has("input")) {
        string str = parser.get<String>("input");
        /* Check if input has only digits, then it should open camera device */
        if (all_of(str.begin(), str.end(), ::isdigit) == false)
          cam.open(str); /* Open video file */
        else
          cam.open(stoi(str)); /* Open camera device */
    }

		ifstream input( labelPath );
		for( string line; getline( input, line ); )
		{
			classes.push_back( line);
		}

		cam_width = cam.get(CAP_PROP_FRAME_WIDTH);
		cam_height = cam.get(CAP_PROP_FRAME_HEIGHT);
    int cnt=0;
    int num_frames = 0;
    double elapsed_time_ns;
    uint32_t totalRunTime;

    /* Start While Loop to process image sequences */
    while (waitKey(1) < 0) {

      /* Start time for FPS Calcuation */
      auto fps_start = std::chrono::high_resolution_clock::now();

      /* Update confidence threshold from the GUI */

      Prob_Threshold = getTrackbarPos(Trackbar_Name, Window_Name) * 0.01f;

      Mat input_image;

      auto success = cam.read(input_image);

      if (!success) {
        cout << "No more frames available!" << endl;
        break;
      }

      num_frames++;

      /* Resize Input Image to fit the SSD MobileNet */
      Mat resized_image;
      resize(input_image, resized_image, Size(300,300));

      /* Allocate Tensors required to run the Forward Pass */
      interpreter->AllocateTensors();

      uchar* input = interpreter->typed_input_tensor<uchar>(0);

      /* Feed input to the model */
      auto image_height=resized_image.rows;
      auto image_width=resized_image.cols;
      auto image_channels=3;

      int number_of_pixels = image_height * image_width * image_channels;
      int base_index = 0;

      /* copy resized_image to input as input tensor */

      memcpy(interpreter->typed_input_tensor<uchar>(0), resized_image.data, resized_image.total() * resized_image.elemSize());

      interpreter->SetAllowFp16PrecisionForFp32(true);

      /* Invoke the Interprter to generate outputs and note the inference time */
      /************************************/
      auto t_start = std::chrono::high_resolution_clock::now();

      interpreter->Invoke();

      auto t_end = std::chrono::high_resolution_clock::now();
      double elapsed_time_ms = std::chrono::duration<double, std::milli>(t_end-t_start).count();
      /************************************/

      /* Perform Post Processing of the outputs from the network */
      post_processing(input_image, interpreter);

      /*********************************************************************
              Display useful statistical information on the image
        *******************************************************************/
      Size labelSize_1;
      int baseLine;

      /* Display Inference Time on the image */
      string label = format("Inference time: %.2f ms", elapsed_time_ms);
      labelSize_1 = getTextSize(label, FONT_HERSHEY_SIMPLEX, 1, 1, &baseLine);
      putText(input_image, label, Point(0, baseLine + labelSize_1.height), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 0, 0), 2);

      /* Average FPS Calcuation */
      auto fps_end = std::chrono::high_resolution_clock::now();
      elapsed_time_ns = std::chrono::duration<double, std::nano>(fps_end-fps_start).count();
      totalRunTime += chrono::duration<double, ratio<1>>(fps_end-fps_start).count();
      double averageFPS = (previous_fps * (num_frames - 1) + (1e+09/(elapsed_time_ns)) ) / num_frames;

      /* Display Average FPS Information on the image */
      label = format("Average FPS: %.2f fps", averageFPS);
      previous_fps = averageFPS;
      Size labelSize_2 = getTextSize(label, FONT_HERSHEY_SIMPLEX, 1, 1, &baseLine);
      labelSize_2.height += labelSize_1.height *  1.5+ baseLine;
      putText(input_image, label, Point(0, labelSize_2.height), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2);

      imshow(Window_Name, input_image);
		}

    return 0;
}

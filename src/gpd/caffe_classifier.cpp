#include <sstream>
#include <fstream>
#include <iostream>
#include <utility>
#include "../../include/gpd/caffe_classifier.h"
#include <algorithm>

CaffeClassifier::CaffeClassifier(const std::string& model_file, const std::string& weights_file, bool use_gpu)
{
  // Initialize Caffe.
  if (use_gpu)
  {
    caffe::Caffe::set_mode(caffe::Caffe::GPU);
  }
  else
  {
    caffe::Caffe::set_mode(caffe::Caffe::CPU);
  }

  // Load pretrained network.
  net_.reset(new caffe::Net<float>(model_file, caffe::TEST));
  net_->CopyTrainedLayersFrom(weights_file);

  input_layer_ = boost::static_pointer_cast < caffe::MemoryDataLayer < float > >(net_->layer_by_name("data"));
}

// sam: utility
bool comp(std::pair<int,float> a, std::pair<int,float> b) {
  return a.second > b.second;
}

std::vector<float> CaffeClassifier::classifyImages(const std::vector<cv::Mat>& image_list)
{
  int batch_size = input_layer_->batch_size();
  int num_iterations = (int) ceil(image_list.size() / (double) batch_size);
  float loss = 0.0;
  std::cout << "# images: " << image_list.size() << ", # iterations: " << num_iterations << ", batch size: " << batch_size << "\n";

  std::vector<float> predictions;

  // Process the images in batches.
  for (int i = 0; i < num_iterations; i++)
  {
    std::vector<cv::Mat>::const_iterator end_it;
    std::vector<cv::Mat> sub_image_list;

    if (i < num_iterations - 1)
    {
      end_it = image_list.begin() + (i + 1) * batch_size;
      sub_image_list.assign(image_list.begin() + i * batch_size, end_it);
    }
    // Fill the batch with empty images to match the required batch size.
    else
    {
      end_it = image_list.end();
      sub_image_list.assign(image_list.begin() + i * batch_size, end_it);
      std::cout << "Adding " << batch_size - sub_image_list.size() << " empty images to batch to match batch size.\n";

      for (int t = sub_image_list.size(); t < batch_size; t++)
      {
        cv::Mat empty_mat(input_layer_->height(), input_layer_->width(), CV_8UC(input_layer_->channels()), cv::Scalar(0.0));
        sub_image_list.push_back(empty_mat);
      }
    }
 
    std::vector<int> label_list;
    label_list.resize(sub_image_list.size());

    for (int j = 0; j < label_list.size(); j++)
    {
      label_list[j] = 0;
    }

    // Classify the batch.
    input_layer_->AddMatVector(sub_image_list, label_list);
    std::vector<caffe::Blob<float>*> results = net_->Forward(&loss);
    std::vector<float> out(results[0]->cpu_data(), results[0]->cpu_data() + results[0]->count());
//    std::cout << "#results: " << results.size() << ", " << results[0]->count() << "\n";

    for (int l = 0; l < results[0]->count() / results[0]-> channels(); l++)
    {
      predictions.push_back(out[2 * l + 1] - out[2 * l]);
//      std::cout << "positive score: " << out[2 * l + 1] << ", negative score: " << out[2 * l] << "\n";
    }
  
// FRI II BWI GROUP STUFF FOLLOWS

    std::cout << "BWI GOT HERE" << std::endl;
    std::vector< std::pair<int,float> > sorting;
    for (int i=0; i<predictions.size(); i++) {
        if(predictions[i] > 0.0) {
            std::cout << predictions[i] << std::endl;
            sorting.push_back(std::make_pair(i,predictions[i]));
        }
    }
    std::sort(sorting.begin(), sorting.end(), comp);

    //ofstream myfile;
    //myfile.open("GOT_HERE.txt");
    //myfile << ""
    cv::FileStorage file ("/home/ladybird/Desktop/bwi_training_grasp_images.yml", cv::FileStorage::APPEND);
    for (int i=0; i<sorting.size() && i<10; i++) {
    /*    file << "score: " << sorting[i].second << "\n";
        for (int j=0; j<sub_image_list[sorting[i].first].rows; j++) {
            const float* Mj = sub_image_list[sorting[i].first].ptr<float>(j);
            for (int k=0; k<sub_image_list[sorting[i].first].cols; k++) {
                file << Mj[k] << " ";
            }
            file << "\n";
        }
    */
        file << "double" << sorting[i].second;
        std::stringstream ss;
        ss << "GraspImage" << i;
        file << ss.str() << sub_image_list[i];
        cv::Mat image = sub_image_list[sorting[i].first];
        file << "image" << image;
            //sub_image_list[i];
    }
    file.release();
  
  }
  return predictions;
}

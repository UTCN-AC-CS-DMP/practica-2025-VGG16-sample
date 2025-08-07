#include <chrono>
#include <iostream>
#include <opencv2/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <vector>

const std::vector<std::string> CIFAR100_LABELS = {
    "apple",    "aquarium_fish", "baby",       "bear",         "beaver",
    "bed",      "bee",           "beetle",     "bicycle",      "bottle",
    "bowl",     "boy",           "bridge",     "bus",          "butterfly",
    "camel",    "can",           "castle",     "caterpillar",  "cattle",
    "chair",    "chimpanzee",    "clock",      "cloud",        "cockroach",
    "couch",    "crab",          "crocodile",  "cup",          "dinosaur",
    "dolphin",  "elephant",      "flatfish",   "forest",       "fox",
    "girl",     "hamster",       "house",      "kangaroo",     "keyboard",
    "lamp",     "lawn_mower",    "leopard",    "lion",         "lizard",
    "lobster",  "man",           "maple_tree", "motorcycle",   "mountain",
    "mouse",    "mushroom",      "oak_tree",   "orange",       "orchid",
    "otter",    "palm_tree",     "pear",       "pickup_truck", "pine_tree",
    "plain",    "plate",         "poppy",      "porcupine",    "possum",
    "rabbit",   "raccoon",       "ray",        "road",         "rocket",
    "rose",     "sea",           "seal",       "shark",        "shrew",
    "skunk",    "skyscraper",    "snail",      "snake",        "spider",
    "squirrel", "streetcar",     "sunflower",  "sweet_pepper", "table",
    "tank",     "telephone",     "television", "tiger",        "tractor",
    "train",    "trout",         "tulip",      "turtle",       "wardrobe",
    "whale",    "willow_tree",   "wolf",       "woman",        "worm"};

std::tuple<std::string, float, double> getClassLabelAndConfidence(
    const cv::Mat& img, cv ::dnn::Net& net) {
  // Convert to float and normalize (same as training)
  cv::Mat blob = cv::dnn::blobFromImage(
      img,
      1.0 / 255.0,  // scale factor
      cv::Size(227, 227),
      cv::Scalar(0.4914 * 255, 0.4822 * 255, 0.4465 * 255),  // mean subtraction
      true,  // swapRB (BGR → RGB)
      false  // don't crop
  );

  // Set the blob as input to the network
  net.setInput(blob);

  auto start = std::chrono::high_resolution_clock::now();

  // Run forward pass
  cv::Mat output = net.forward();

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> duration = end - start;

  // Find the class with the highest score
  cv::Point classIdPoint;
  double confidence;
  minMaxLoc(output, nullptr, &confidence, nullptr, &classIdPoint);
  int predicted_class = classIdPoint.x;
  std::string class_label = CIFAR100_LABELS[predicted_class];
  return {class_label, static_cast<float>(confidence), duration.count()};
}

void printClassLabelAndConfidence(const std::string& expectedLabel,
                                  const std::string& fileName,
                                  cv ::dnn::Net& net) {
  std::string imagePath = ASSETS_DIR "pictures/" + fileName + ".jpg";
  cv::Mat image = cv::imread(imagePath);
  if (image.empty()) {
    std::cerr << "Could not load the image: " << imagePath << std::endl;
  } else {
    // Resize to 227x227 as expected by the model
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(227, 227));
    auto [classLabel, confidence, duration] =
        getClassLabelAndConfidence(resized, net);
    std::cout << "Expected animal: " << expectedLabel << std::endl;
    std::cout << "Processing time: " << duration << " ms" << std::endl;
    std::cout << "Predicted animal: " << classLabel << std::endl;
    std::cout << "Confidence: " << confidence << std::endl;
  }
}

int main() {
  // Load the ONNX model
  std::string modelPath = ASSETS_DIR "models/vgg16_cifar100.onnx";
  cv::dnn::Net net = cv::dnn::readNetFromONNX(modelPath);
  if (net.empty()) {
    std::cerr << "Could not load the model: " << modelPath << std::endl;
    return 1;
  }

  // Load and preprocess the input images
  std::vector<std::pair<std::string, std::string>> images = {
      {"dog", "golden_retriever"},
      {"cat", "orange_cat"},
      {"rabbit", "european_rabbit"},
      {"horse", "horse"}};

  for (const auto& [label, path] : images) {
    printClassLabelAndConfidence(label, path, net);
    std::cout << "----------------------------------------" << std::endl;
  }

  return 0;
}
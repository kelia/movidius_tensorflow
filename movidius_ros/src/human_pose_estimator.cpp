#include <algorithm>
#include <string>
#include <vector>

#include <opencv2/imgproc/imgproc.hpp>

#include <movidius_ros/common.h>

#include "movidius_ros/human_pose_estimator.h"

namespace human_pose_estimation {

HumanPoseEstimator::HumanPoseEstimator(const std::string &modelPath,
                                       const std::string &targetDeviceName,
                                       bool enablePerformanceReport)
    : minJointsNumber(3),
      stride(8),
      inputLayerSize(-1, -1),
      enablePerformanceReport(enablePerformanceReport),
      modelPath(modelPath) {
  plugin = InferenceEngine::PluginDispatcher({"../../../lib/intel64", ""})
      .getPluginByDevice(targetDeviceName);
  if (enablePerformanceReport) {
    plugin.SetConfig({{InferenceEngine::PluginConfigParams::KEY_PERF_COUNT,
                       InferenceEngine::PluginConfigParams::YES}});
  }
  netReader.ReadNetwork(modelPath);
  std::string binFileName = fileNameNoExt(modelPath) + ".bin";
  netReader.ReadWeights(binFileName);
  network = netReader.getNetwork();
  InferenceEngine::InputInfo::Ptr inputInfo = network.getInputsInfo().begin()->second;
  std::cout << "network.getInputsInfo().begin()->second: " << network.getInputsInfo().begin()->second << std::endl;
  std::cout << "Input dimension: "
            << inputInfo->getTensorDesc().getDims()[0] << " | "
            << inputInfo->getTensorDesc().getDims()[1] << " | "
            << inputInfo->getTensorDesc().getDims()[2] << " | "
            << inputInfo->getTensorDesc().getDims()[3] << std::endl;

  inputLayerSize = cv::Size(inputInfo->getTensorDesc().getDims()[3], inputInfo->getTensorDesc().getDims()[2]);
  InferenceEngine::OutputsDataMap outputInfo = network.getOutputsInfo();
  auto outputBlobsIt = outputInfo.begin();
  pafsBlobName = outputBlobsIt->first;
  executableNetwork = plugin.LoadNetwork(network, {});
  request = executableNetwork.CreateInferRequest();
}

std::vector<GatePose> HumanPoseEstimator::estimate(const cv::Mat &image) {
  CV_Assert(image.type() == CV_8UC3);

  cv::Size imageSize = image.size();
  InferenceEngine::Blob::Ptr input = request.GetBlob(network.getInputsInfo().begin()->first);

  auto data = input->buffer().as<InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP32>::value_type *>();

  // copy cv image
  //convert from bgr to rgb
  cv::Mat image_copy1 = image.clone();
  cv::cvtColor(image, image_copy1, CV_BGR2RGB);

  image_copy1.convertTo(image_copy1, CV_32FC3);

  float beta = 1.0f / 255.0f;
  cv::Mat image_copy = image_copy1 * beta;
  size_t num_channels = input->dims()[2];
  size_t image_size = input->dims()[1] * input->dims()[0];


  /** Iterate over all pixel in image (b,g,r) **/
  for (size_t pid = 0; pid < image_size; pid++) {
    /** Iterate over all channels **/
    for (size_t ch = 0; ch < num_channels; ++ch) {
      /**          [images stride + channels stride + pixel id ] all in bytes            **/
      float *matData = (float *) image_copy.data;
      data[ch * image_size + pid] = matData[pid * num_channels + ch];
    }
  }

  request.Infer();

  InferenceEngine::Blob::Ptr pafsBlob = request.GetBlob(pafsBlobName);

  const float *detection =
      static_cast<InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP32>::value_type *>(pafsBlob->buffer());

  float r = detection[0];
  float theta = detection[1];
  float phi = detection[2];
  float yaw = detection[3];
  float var_r = detection[4];
  float var_theta = detection[5];
  float var_phi = detection[6];
  float var_yaw = detection[7];


  std::vector<GatePose> poses;
  poses.emplace_back(GatePose(r, theta, phi, yaw, var_r, var_theta, var_phi, var_yaw));

  return poses;
}

void HumanPoseEstimator::preprocess(const cv::Mat &image, float *buffer) const {
//  cv::Mat resizedImage;
//  double scale = inputLayerSize.height / static_cast<double>(image.rows);
//  std::cout << "scale = " << scale << std::endl;
//  cv::resize(image, resizedImage, cv::Size(), scale, scale, cv::INTER_CUBIC);
//  cv::Mat paddedImage;
//  cv::copyMakeBorder(image, paddedImage, pad(0), pad(2), pad(1), pad(3),
//                     cv::BORDER_CONSTANT, meanPixel);
//  std::cout << "pad: " << pad << std::endl;
//  cv::copyMakeBorder(resizedImage, paddedImage, pad(0), pad(2), pad(1), pad(3),
//                     cv::BORDER_CONSTANT, meanPixel);
  std::vector<cv::Mat> planes(3);
  cv::split(image, planes);
//  cv::split(paddedImage, planes);
  for (size_t pId = 0; pId < planes.size(); pId++) {
    cv::Mat dst(inputLayerSize.height, inputLayerSize.width, CV_32FC1,
                reinterpret_cast<void *>(
                    buffer + pId * inputLayerSize.area()));
//    cv::Mat dst(inputLayerSize.height, inputLayerSize.width, CV_8UC3,
//                reinterpret_cast<void *>(
//                    buffer + pId * inputLayerSize.area()));

    //cv::Mat test(rows, cols, type, data);
//    std::cout << (int) planes[pId].at<uchar>(100, 100) << std::endl;
    planes[pId].convertTo(dst, CV_32FC1);
//    std::cout << "test: " << dst.at<float>(100, 100) / 255.0 << std::endl;
//    std::cout << "dst: " << dst.at<float>(100, 100) << std::endl;
//    planes[pId].convertTo(dst, CV_8UC3);
  }
}

//bool HumanPoseEstimator::inputWidthIsChanged(const cv::Size &imageSize) {
//  double scale = inputLayerSize.height / static_cast<double>(imageSize.height);
//  cv::Size scaledSize(cvRound(imageSize.width * scale),
//                      cvRound(imageSize.height * scale));
//  cv::Size scaledImageSize(std::max(scaledSize.width, inputLayerSize.height),
//                           inputLayerSize.height);
//  int minHeight = std::min(scaledImageSize.height, scaledSize.height);
//  scaledImageSize.width = std::ceil(
//      scaledImageSize.width / static_cast<float>(stride)) * stride;
//  pad(0) = std::floor((scaledImageSize.height - minHeight) / 2.0);
//  pad(1) = std::floor((scaledImageSize.width - scaledSize.width) / 2.0);
//  pad(2) = scaledImageSize.height - minHeight - pad(0);
//  pad(3) = scaledImageSize.width - scaledSize.width - pad(1);
//  if (scaledSize.width == (inputLayerSize.width - pad(1) - pad(3))) {
//    return false;
//  }
//
//  inputLayerSize.width = scaledImageSize.width;
//  return true;
//}

std::string HumanPoseEstimator::type2str(int type) {
  std::string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch (depth) {
    case CV_8U: r = "8U";
      break;
    case CV_8S: r = "8S";
      break;
    case CV_16U: r = "16U";
      break;
    case CV_16S: r = "16S";
      break;
    case CV_32S: r = "32S";
      break;
    case CV_32F: r = "32F";
      break;
    case CV_64F: r = "64F";
      break;
    default: r = "User";
      break;
  }

  r += "C";
  r += (chans + '0');

  return r;
}

HumanPoseEstimator::~HumanPoseEstimator() {
  if (enablePerformanceReport) {
    std::cout << "Performance counts for " << modelPath << std::endl << std::endl;
    printPerformanceCounts(request.GetPerformanceCounts(), std::cout, false);
  }
}

/**
* @brief Sets image data stored in cv::Mat object to a given Blob object.
* @param orig_image - given cv::Mat object with an image data.
* @param blob - Blob object which to be filled by an image data.
* @param batchIndex - batch index of an image inside of the blob.
*/
template<typename T>
void HumanPoseEstimator::matU8ToBlob(const cv::Mat &orig_image, InferenceEngine::Blob::Ptr &blob) {
  InferenceEngine::SizeVector blobSize = blob->getTensorDesc().getDims();
  const size_t width = blobSize[3];
  const size_t height = blobSize[2];
  const size_t channels = blobSize[1];
  T *blob_data = blob->buffer().as<T *>();

  int batchIndex = 0;
  cv::Mat resized_image(orig_image);
  if (width != orig_image.size().width || height != orig_image.size().height) {
    cv::resize(orig_image, resized_image, cv::Size(width, height));
  }

  int batchOffset = batchIndex * width * height * channels;

  for (size_t c = 0; c < channels; c++) {
    for (size_t h = 0; h < height; h++) {
      for (size_t w = 0; w < width; w++) {
        blob_data[batchOffset + c * width * height + h * width + w] =
            resized_image.at<cv::Vec3b>(h, w)[c];
      }
    }
  }
}

/**
 * @brief Wraps data stored inside of a passed cv::Mat object by new Blob pointer.
 * @note: No memory allocation is happened. The blob just points to already existing
 *        cv::Mat data.
 * @param mat - given cv::Mat object with an image data.
 * @return resulting Blob pointer.
 */
InferenceEngine::Blob::Ptr HumanPoseEstimator::wrapMat2Blob(const cv::Mat &mat) {
  size_t channels = mat.channels();
  size_t height = mat.size().height;
  size_t width = mat.size().width;

  size_t strideH = mat.step.buf[0];
  size_t strideW = mat.step.buf[1];

  bool is_dense =
      strideW == channels &&
          strideH == channels * width;

  if (!is_dense)
    THROW_IE_EXCEPTION
        << "Doesn't support conversion from not dense cv::Mat";

  InferenceEngine::TensorDesc tDesc(InferenceEngine::Precision::U8,
                                    {1, channels, height, width},
                                    InferenceEngine::Layout::NHWC);

  return InferenceEngine::make_shared_blob<uint8_t>(tDesc, mat.data);
}

}  // namespace human_pose_estimation

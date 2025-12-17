#include <torch/torch.h>
#include <torch/script.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <thread>
#include <mutex>
#include <atomic>

namespace fs = std::filesystem;

struct Result {
    std::string image_path;
    std::string filename;
    std::string true_label;
    std::string predicted_label;
    float confidence;
    std::string error;
};

class DeepfakeDetector {
private:
    torch::jit::script::Module model;
    torch::Device device;
    std::mutex mtx;

public:
    DeepfakeDetector(const std::string& model_path, bool use_cuda = true)
        : device(use_cuda && torch::cuda::is_available() ? torch::kCUDA : torch::kCPU) {

        std::cout << "Loading model from: " << model_path << std::endl;

        try {
            // Load the traced/scripted model
            model = torch::jit::load(model_path + "/traced_model.pt");
            model.to(device);
            model.eval();

            std::cout << "✓ Model loaded on: "
                      << (device.is_cuda() ? "CUDA" : "CPU") << std::endl;
        } catch (const c10::Error& e) {
            std::cerr << "Error loading model: " << e.what() << std::endl;
            throw;
        }
    }

    torch::Tensor preprocess(const cv::Mat& image) {
        // Resize to 256
        cv::Mat resized;
        cv::resize(image, resized, cv::Size(256, 256));

        // Center crop to 224x224
        int crop_x = (resized.cols - 224) / 2;
        int crop_y = (resized.rows - 224) / 2;
        cv::Rect roi(crop_x, crop_y, 224, 224);
        cv::Mat cropped = resized(roi);

        // Convert BGR to RGB
        cv::Mat rgb;
        cv::cvtColor(cropped, rgb, cv::COLOR_BGR2RGB);

        // Convert to float and normalize [0, 255] -> [0, 1]
        rgb.convertTo(rgb, CV_32FC3, 1.0 / 255.0);

        // Normalize with mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
        cv::Mat normalized;
        cv::subtract(rgb, cv::Scalar(0.5, 0.5, 0.5), normalized);
        cv::divide(normalized, cv::Scalar(0.5, 0.5, 0.5), normalized);

        // Convert to tensor [H, W, C] -> [C, H, W]
        torch::Tensor tensor = torch::from_blob(
            normalized.data,
            {224, 224, 3},
            torch::kFloat32
        ).clone();

        tensor = tensor.permute({2, 0, 1}); // [H, W, C] -> [C, H, W]
        tensor = tensor.unsqueeze(0);       // Add batch dimension [1, C, H, W]

        return tensor;
    }

    Result predict_image(const std::string& image_path) {
        Result result;
        result.image_path = image_path;
        result.filename = fs::path(image_path).filename().string();

        try {
            // Load image
            cv::Mat image = cv::imread(image_path);
            if (image.empty()) {
                throw std::runtime_error("Failed to load image");
            }

            // Preprocess
            torch::Tensor input_tensor = preprocess(image);
            input_tensor = input_tensor.to(device);

            // Convert to half precision (float16)
            if (device.is_cuda()) {
                input_tensor = input_tensor.to(torch::kHalf);
            }

            // Inference
            torch::NoGradGuard no_grad;
            std::vector<torch::jit::IValue> inputs;
            inputs.push_back(input_tensor);

            auto output = model.forward(inputs).toTensor();

            // Get probabilities
            auto probs = torch::softmax(output, 1);
            auto [max_prob, predicted_class] = torch::max(probs, 1);

            int pred_class = predicted_class.item<int>();
            float confidence = max_prob.item<float>();

            // Map class to label (swap if needed based on your model)
            // Assuming: 0 = Realism, 1 = Deepfake (adjust based on your model)
            std::string label = (pred_class == 0) ? "Deepfake" : "Realism";

            result.predicted_label = label;
            result.confidence = confidence;

        } catch (const std::exception& e) {
            result.predicted_label = "Error";
            result.confidence = 0.0f;
            result.error = e.what();
        }

        return result;
    }

    std::string get_true_label(const std::string& image_path) {
        std::string parent_folder = fs::path(image_path).parent_path().filename().string();
        std::transform(parent_folder.begin(), parent_folder.end(),
                       parent_folder.begin(), ::tolower);

        if (parent_folder.find("fake") != std::string::npos ||
            parent_folder.find("deepfake") != std::string::npos) {
            return "Deepfake";
        } else if (parent_folder.find("real") != std::string::npos) {
            return "Realism";
        }
        return "Unknown";
    }
};

std::vector<std::string> get_image_files(const std::string& dataset_path) {
    std::vector<std::string> image_files;

    for (const auto& entry : fs::recursive_directory_iterator(dataset_path)) {
        if (entry.is_regular_file()) {
            std::string ext = entry.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

            if (ext == ".jpg" || ext == ".jpeg" || ext == ".png") {
                image_files.push_back(entry.path().string());
            }
        }
    }

    return image_files;
}

void print_results(const std::vector<Result>& results) {
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "RESULTS" << std::endl;
    std::cout << std::string(80, '=') << std::endl;

    int correct = 0;
    int total = 0;

    for (const auto& result : results) {
        if (result.predicted_label != "Error" && result.true_label != "Unknown") {
            std::cout << "\nFile: " << result.filename << std::endl;
            std::cout << "  True Label:      " << result.true_label << std::endl;
            std::cout << "  Predicted Label: " << result.predicted_label << std::endl;
            std::cout << "  Confidence:      " << std::fixed << std::setprecision(4)
                      << result.confidence << std::endl;

            if (result.true_label == result.predicted_label) {
                std::cout << "  Status: ✓ CORRECT" << std::endl;
                correct++;
            } else {
                std::cout << "  Status: ✗ INCORRECT" << std::endl;
            }
            total++;
        } else if (result.predicted_label == "Error") {
            std::cout << "\nFile: " << result.filename << std::endl;
            std::cout << "  Error: " << result.error << std::endl;
        }
    }

    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "SUMMARY" << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    std::cout << "Total images processed: " << results.size() << std::endl;
    std::cout << "Valid predictions: " << total << std::endl;

    if (total > 0) {
        float accuracy = static_cast<float>(correct) / total;
        std::cout << "Correct predictions: " << correct << std::endl;
        std::cout << "Accuracy: " << std::fixed << std::setprecision(4)
                  << accuracy << std::endl;
    }
}

int main() {
    // Paths (relative to project root)
    fs::path project_root = fs::path(__FILE__).parent_path().parent_path();
    fs::path model_path = project_root / "model" / "models" / "deepfake-detector-v2";
    fs::path dataset_path = project_root / "data" / "Dataset" / "Test";

    std::cout << "All imports successful!" << std::endl;

    // Initialize detector
    DeepfakeDetector detector(model_path.string(), true);

    // Get image files
    std::vector<std::string> img_files = get_image_files(dataset_path.string());
    std::cout << "\nFound " << img_files.size() << " images in test set" << std::endl;

    // Process images
    std::vector<Result> results;
    results.reserve(img_files.size());

    std::mutex results_mtx;
    std::atomic<size_t> processed(0);

    const size_t num_threads = std::thread::hardware_concurrency();
    std::cout << "Processing with " << num_threads << " threads..." << std::endl;

    // Parallel processing
    auto process_batch = [&](size_t start, size_t end) {
        for (size_t i = start; i < end; ++i) {
            Result result = detector.predict_image(img_files[i]);
            result.true_label = detector.get_true_label(img_files[i]);

            {
                std::lock_guard<std::mutex> lock(results_mtx);
                results.push_back(result);
            }

            size_t count = ++processed;
            if (count % 100 == 0) {
                std::cout << "Processed: " << count << "/" << img_files.size()
                          << std::endl;
            }
        }
    };

    // Launch threads
    std::vector<std::thread> threads;
    size_t batch_size = img_files.size() / num_threads;

    for (size_t i = 0; i < num_threads; ++i) {
        size_t start = i * batch_size;
        size_t end = (i == num_threads - 1) ? img_files.size() : (i + 1) * batch_size;
        threads.emplace_back(process_batch, start, end);
    }

    // Wait for completion
    for (auto& thread : threads) {
        thread.join();
    }

    std::cout << "\nParallel processing complete." << std::endl;

    // Print results to console
    print_results(results);

    return 0;
}

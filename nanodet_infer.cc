
#include <cstddef>
#include <iostream>
#include <memory>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include  "my_interface.h"
#include <chrono>
#include <vector>
#include <npp.h>

#define INPUT_W 320
#define INPUT_H 320
#define OUTPUT_SHAPE  (INPUT_W/320)*(INPUT_W/320)*2100
#define IsPadding 1
#define NUM_CLASS 1
#define NMS_THRESH 0.6
#define CONF_THRESH 0.3

char* output_name = "803";
char* trt_model_path = "../models/nanodet_int8.trt";
std::string test_img = "../test_imgs/1c9ff24d-fadf-3984-9c83-3d1937c0526e.jpg";

std::vector<int> strides = {8,16,32};
std::vector<float> img_mean = {103.53, 116.28, 123.675};
std::vector<float> img_std = { 57.375,  57.12,  58.395 };


using namespace cv;
using namespace std;

struct Bbox{
    float x;
    float y;
    float w;
    float h;
    float prob;
    int classes;
};
cv::Mat refer_matrix;
int refer_rows = 0;
int refer_cols = 3;
void GenerateReferMatrix() {
    int index = 0;
    refer_matrix = cv::Mat(refer_rows, refer_cols, CV_32FC1);
    for (const int &stride : strides) {
        for (int h = 0; h < INPUT_H / stride; h++)
            for (int w = 0; w < INPUT_W / stride; w++) {
                auto *row = refer_matrix.ptr<float>(index);
                row[0] = float((2 * w + 1) * stride - 1) / 2;
                row[1] = float((2 * h + 1) * stride - 1) / 2;
                row[2] = stride;
                index += 1;
            }
    }
}
float IOUCalculate(const Bbox &det_a, const Bbox &det_b) {
    cv::Point2f center_a(det_a.x, det_a.y);
    cv::Point2f center_b(det_b.x, det_b.y);
    cv::Point2f left_up(std::min(det_a.x - det_a.w / 2, det_b.x - det_b.w / 2),
                        std::min(det_a.y - det_a.h / 2, det_b.y - det_b.h / 2));
    cv::Point2f right_down(std::max(det_a.x + det_a.w / 2, det_b.x + det_b.w / 2),
                           std::max(det_a.y + det_a.h / 2, det_b.y + det_b.h / 2));
    float distance_d = (center_a - center_b).x * (center_a - center_b).x + (center_a - center_b).y * (center_a - center_b).y;
    float distance_c = (left_up - right_down).x * (left_up - right_down).x + (left_up - right_down).y * (left_up - right_down).y;
    float inter_l = det_a.x - det_a.w / 2 > det_b.x - det_b.w / 2 ? det_a.x - det_a.w / 2 : det_b.x - det_b.w / 2;
    float inter_t = det_a.y - det_a.h / 2 > det_b.y - det_b.h / 2 ? det_a.y - det_a.h / 2 : det_b.y - det_b.h / 2;
    float inter_r = det_a.x + det_a.w / 2 < det_b.x + det_b.w / 2 ? det_a.x + det_a.w / 2 : det_b.x + det_b.w / 2;
    float inter_b = det_a.y + det_a.h / 2 < det_b.y + det_b.h / 2 ? det_a.y + det_a.h / 2 : det_b.y + det_b.h / 2;
    if (inter_b < inter_t || inter_r < inter_l)
        return 0;
    float inter_area = (inter_b - inter_t) * (inter_r - inter_l);
    float union_area = det_a.w * det_a.h + det_b.w * det_b.h - inter_area;
    if (union_area == 0)
        return 0;
    else
        return inter_area / union_area - distance_d / distance_c;
}
void NmsDetect(std::vector<Bbox> &detections) {
    sort(detections.begin(), detections.end(), [=](const Bbox &left, const Bbox &right) {
        return left.prob > right.prob;
    });

    for (int i = 0; i < (int)detections.size(); i++)
        for (int j = i + 1; j < (int)detections.size(); j++)
        {
            float iou = IOUCalculate(detections[i], detections[j]);
            if (iou > NMS_THRESH)
                detections[j].prob = 0;
        }

    detections.erase(std::remove_if(detections.begin(), detections.end(), [](const Bbox &det)
    { return det.prob == 0; }), detections.end());
}

std::vector<Bbox> postProcess(const cv::Mat &src_img,
                              float *output, const int &outSize) {
    std::vector<Bbox> result;
    float *out = output;
    float ratio = std::max(float(src_img.cols) / float(INPUT_W), float(src_img.rows) / float(INPUT_H));
    cv::Mat result_matrix = cv::Mat(refer_rows, NUM_CLASS + 4, CV_32FC1, out);
    for (int row_num = 0; row_num < refer_rows; row_num++) {
        Bbox box;
        auto *row = result_matrix.ptr<float>(row_num);
        auto max_pos = std::max_element(row + 4, row + NUM_CLASS + 4);
        box.prob = row[max_pos - row];
        if (box.prob < CONF_THRESH)
            continue;
        box.classes = max_pos - row - 4;
        auto *anchor = refer_matrix.ptr<float>(row_num);
        box.x = (anchor[0] - row[0] * anchor[2] + anchor[0] + row[2] * anchor[2]) / 2 * ratio;
        box.y = (anchor[1] - row[1] * anchor[2] + anchor[1] + row[3] * anchor[2]) / 2 * ratio;
        box.w = (row[2] + row[0]) * anchor[2] * ratio;
        box.h = (row[3] + row[1]) * anchor[2] * ratio;
        result.push_back(box);
    }
        NmsDetect(result);
    return result;
}


cv::Mat renderBoundingBox(cv::Mat image, const std::vector<Bbox> &bboxes){
    //float scale = std::min(static_cast<float>(INPUT_W) / static_cast<float>(image.cols), static_cast<float>(INPUT_H) / static_cast<float>(image.rows));
    for(const auto &rect : bboxes)
    {
        cv::Rect rst(rect.x - rect.w / 2, rect.y - rect.h / 2, rect.w, rect.h);
        cv::rectangle(image, rst, cv::Scalar(255, 204,0), 2, cv::LINE_8, 0);
        //cv::rectangle(image, cv::Point(rect.x - rect.w / 2, rect.y - rect.h / 2), cv::Point(rect.x + rect.w / 2, rect.y + rect.h / 2), cv::Scalar(255, 204,0), 3);
        cv::putText(image, std::to_string(rect.prob), cv::Point(rect.x - rect.w / 2, rect.y - rect.h / 2), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,204,255));
    }
    return image;
}


cv::Mat preprocess_img(cv::Mat& img) {
    int w, h, x, y;
    float r_w = INPUT_W / (img.cols*1.0);
    float r_h = INPUT_H / (img.rows*1.0);
    if (r_h > r_w) {
        w = INPUT_W;
        h = r_w * img.rows;
        x = 0;
        y = (INPUT_H - h) / 2;
    } else {
        w = r_h * img.cols;
        h = INPUT_H;
        x = (INPUT_W - w) / 2;
        y = 0;
    }
    cv::Mat re(h, w, CV_8UC3);
    cv::resize(img, re, re.size(), 0, 0, cv::INTER_LINEAR);
    cv::Mat out(INPUT_H, INPUT_W, CV_8UC3, cv::Scalar(128, 128, 128));
    re.copyTo(out(cv::Rect(x, y, re.cols, re.rows)));
    return out;
}

void cudaResize(cv::Mat &image, cv::Mat &rsz_img)
{
    int outsize = rsz_img.cols * rsz_img.rows * sizeof(uchar3);

    int inwidth = image.cols;
    int inheight = image.rows;
    int memSize = inwidth * inheight * sizeof(uchar3);

    NppiSize srcsize = {inwidth, inheight};
    NppiRect srcroi  = {0, 0, inwidth, inheight};
    NppiSize dstsize = {rsz_img.cols, rsz_img.rows};
    NppiRect dstroi  = {0, 0, rsz_img.cols, rsz_img.rows};

    uchar3* d_src = NULL;
    uchar3* d_dst = NULL;
    cudaMalloc((void**)&d_src, memSize);
    cudaMalloc((void**)&d_dst, outsize);
    cudaMemcpy(d_src, image.data, memSize, cudaMemcpyHostToDevice);

    // nvidia npp 图像处理
    nppiResize_8u_C3R( (Npp8u*)d_src, inwidth * 3, srcsize, srcroi,
                       (Npp8u*)d_dst, rsz_img.cols * 3, dstsize, dstroi,
                       NPPI_INTER_LINEAR );


    cudaMemcpy(rsz_img.data, d_dst, outsize, cudaMemcpyDeviceToHost);

    cudaFree(d_src);
    cudaFree(d_dst);
}

std::vector<float> prepareImage(cv::Mat &src_img) {
    std::vector<float> result(INPUT_W * INPUT_H * 3);
    float *data = result.data();
    float ratio = float(INPUT_W) / float(src_img.cols) < float(INPUT_H) / float(src_img.rows) ? float(INPUT_W) / float(src_img.cols) : float(INPUT_H) / float(src_img.rows);
    cv::Mat flt_img = cv::Mat::zeros(cv::Size(INPUT_W, INPUT_H), CV_8UC3);
    cv::Mat rsz_img = cv::Mat::zeros(cv::Size(src_img.cols*ratio, src_img.rows*ratio), CV_8UC3);
    //auto pr_start = std::chrono::high_resolution_clock::now();
    //cv::resize(src_img, rsz_img, cv::Size(), ratio, ratio);
    cudaResize(src_img, rsz_img);
    //auto pr_end = std::chrono::high_resolution_clock::now();
    //auto po_ms = std::chrono::duration<float, std::milli>(pr_end - pr_start).count();
    rsz_img.copyTo(flt_img(cv::Rect(0, 0, rsz_img.cols, rsz_img.rows)));
    flt_img.convertTo(flt_img, CV_32FC3);

    //auto pr_end = std::chrono::high_resolution_clock::now();
    //std::cout << "********** " << po_ms << " ms." << "********** " << std::endl;


    //HWC TO CHW
    //auto pr_start = std::chrono::high_resolution_clock::now();
    int channelLength = INPUT_W * INPUT_H;
    std::vector<cv::Mat> split_img = {
            cv::Mat(INPUT_W, INPUT_H, CV_32FC1, data + channelLength * 2),
            cv::Mat(INPUT_W, INPUT_H, CV_32FC1, data + channelLength),
            cv::Mat(INPUT_W, INPUT_H, CV_32FC1, data)
    };
    //auto pr_end = std::chrono::high_resolution_clock::now();
    //auto po_ms = std::chrono::duration<float, std::milli>(pr_end - pr_start).count();
    //std::cout << "********** " << po_ms << " ms." << "********** " << std::endl;


    auto pr_start = std::chrono::high_resolution_clock::now();
    cv::split(flt_img, split_img);
    for (int i = 0; i < 3; i++) {
            split_img[i] = (split_img[i] - img_mean[i]) / img_std[i];
    }
    auto pr_end = std::chrono::high_resolution_clock::now();

    auto po_ms = std::chrono::duration<float, std::milli>(pr_end - pr_start).count();
    std::cout << "********** " << po_ms << " ms." << "********** " << std::endl;
    return result;
}


/////
int main(int argc, const char *argv[]) {
    for (const int &stride : strides) {
        refer_rows += INPUT_W * INPUT_H / stride / stride;
    }
    GenerateReferMatrix();
    float total = 0, ms, pr_ms, po_ms;
    int test_echo = 20;

    // 创建输入输出tensor结构体
    tensor_params_array_t in_tensor_params_ar = {0};
    tensor_params_array_t out_tensor_params_ar = {0};
    tensor_array_t *input_tensor_array = NULL;
    tensor_array_t *ouput_tensor_array = NULL;

    /****************** */
    // 定义输入tensor
    in_tensor_params_ar.nArraySize = 1;
    in_tensor_params_ar.pTensorParamArray = (tensor_params_t *) malloc(
            in_tensor_params_ar.nArraySize * sizeof(tensor_params_t));
    memset(in_tensor_params_ar.pTensorParamArray, 0, in_tensor_params_ar.nArraySize * sizeof(tensor_params_t));

    tensor_params_t *cur_in_tensor_params = in_tensor_params_ar.pTensorParamArray;

    // 第一个输入tensor
    cur_in_tensor_params[0].nDims = 4;
    cur_in_tensor_params[0].type = DT_FLOAT;
    cur_in_tensor_params[0].pShape[0] = 1; //batch size can't set to -1
    cur_in_tensor_params[0].pShape[1] = 3;
    cur_in_tensor_params[0].pShape[2] = INPUT_W;
    cur_in_tensor_params[0].pShape[3] = INPUT_H;
    strcpy(cur_in_tensor_params[0].aTensorName, "input.1");
    cur_in_tensor_params[0].tensorMemoryType = CPU_MEM_ALLOC;

    /*************** */
    // 定义输出tensor
    out_tensor_params_ar.nArraySize = 1;
    out_tensor_params_ar.pTensorParamArray = (tensor_params_t *) malloc(
            out_tensor_params_ar.nArraySize * sizeof(tensor_params_t));
    memset(out_tensor_params_ar.pTensorParamArray, 0, out_tensor_params_ar.nArraySize * sizeof(tensor_params_t));

    tensor_params_t *cur_out_tensor_params = out_tensor_params_ar.pTensorParamArray;

    cur_out_tensor_params[0].nDims = 3;
    cur_out_tensor_params[0].type = DT_FLOAT;
    cur_out_tensor_params[0].pShape[0] = 1;
    cur_out_tensor_params[0].pShape[1] = OUTPUT_SHAPE;
    cur_out_tensor_params[0].pShape[2] = 4+NUM_CLASS;
    cur_out_tensor_params[0].tensorMemoryType = CPU_MEM_ALLOC;
    strcpy(cur_out_tensor_params[0].aTensorName, output_name);


    // 初始化输入输出结构体，分配内存
    if (my_init_tensors(&in_tensor_params_ar, &out_tensor_params_ar,
                        &input_tensor_array, &ouput_tensor_array) != MY_SUCCESS) {
        printf("Open Internal memory error!\n");
    }

    //===================obtain Handle=========================================
    model_params_t tModelParam = {0}; //model input parameter
    model_handle_t tModelHandle = {0};

    strcpy(tModelParam.visibleCard, "0");
    tModelParam.gpu_id = 0; //GPU 0
    tModelParam.bIsCipher = FALSE;
    tModelParam.maxBatchSize = 1;
//    strcpy(tModelParam.model_path, "../models/TRT_ssd_mobilenet_v2_coco.trt");
    strcpy(tModelParam.model_path, trt_model_path);
//    strcpy(tModelParam.model_path, "../models/");

//  tModelParam.bIsCipher = TRUE;
//  tModelParam.encStartPoint = 340;
//  tModelParam.encLength = 5000;
//  strcpy(tModelParam.model_path, "models/encrpy_model");

    //call API open model
    if (my_load_model(&tModelParam,
                      input_tensor_array,
                      ouput_tensor_array,
                      &tModelHandle) != MY_SUCCESS) {
        printf("Open model error!\n");
    }
    std::cout << "Load model sucess\n";


    string file_name = test_img;
    //string file_name = "/home/willer/calibration_data/2bb75da9-331a-3d4f-96d0-5817ae6aed80.jpg";
    tensor_t *cur_input_tensor_image = &(input_tensor_array->pTensorArray[0]);

    cv::Mat cImage;
    cImage = cv::imread(file_name);
    std::cout << "Read img finished!\n";
    cv::Mat showImage = cImage.clone();


//    static float data[3 * INPUT_H * INPUT_W];

//    cv::Mat pre_img = preprocess_img(cImage);
//    std::cout << "preprocess_img finished!\n";
//    int i = 0;
//    for (int row = 0; row < INPUT_H; ++row) {
//        uchar* uc_pixel = pre_img.data + row * pre_img.step;
//        for (int col = 0; col < INPUT_W; ++col) {
//            data[i] = (float)uc_pixel[2] / 255.0;
//            data[i + INPUT_H * INPUT_W] = (float)uc_pixel[1] / 255.0;
//            data[i + 2 * INPUT_H * INPUT_W] = (float)uc_pixel[0] / 255.0;
//            uc_pixel += 3;
//            ++i;
//        }
//    }
    auto pr_start = std::chrono::high_resolution_clock::now();
    vector<float> pr_img = prepareImage(cImage);
    auto pr_end = std::chrono::high_resolution_clock::now();
    pr_ms = std::chrono::duration<float, std::milli>(pr_end - pr_start).count();

    memcpy((float *) (cur_input_tensor_image->pValue),
           pr_img.data(), 3 * INPUT_H * INPUT_W * sizeof(float));


    printf("----->memcpy data is success......\n");
    for (int j = 0; j < test_echo; ++j) {
        auto t_start = std::chrono::high_resolution_clock::now();

        my_inference_tensors(&tModelHandle);

        auto t_end = std::chrono::high_resolution_clock::now();
        ms = std::chrono::duration<float, std::milli>(t_end - t_start).count();
        total += ms;
        std::cout << "[ " << j << " ] " << ms << " ms." << std::endl;
    }

    total /= test_echo;
    std::cout << "Average over " << test_echo << " runs is " << total << " ms." << std::endl;

    tensor_t *cur_output_tensor = &(ouput_tensor_array->pTensorArray[0]);
    float * output = static_cast<float *>(cur_output_tensor->pValue);

    int outSize = cur_output_tensor->pTensorInfo->nElementSize;
    std::cout << "outSize:" << outSize << std::endl;

    auto po_start = std::chrono::high_resolution_clock::now();
    vector<Bbox> bboxes = postProcess(showImage, output, outSize);
    auto po_end = std::chrono::high_resolution_clock::now();
    po_ms = std::chrono::duration<float, std::milli>(po_end - po_start).count();


    showImage = renderBoundingBox(showImage, bboxes);
    cv::imwrite("final.jpg", showImage);

    //std::cout << "prepareImage " << " runs is " << pr_ms << " ms." << std::endl;
    //std::cout << "postProcess " << " runs is " << po_ms << " ms." << std::endl;

    my_deinit_tensors(input_tensor_array, ouput_tensor_array);

    my_release_model(&tModelHandle);

    std::cout << "complete!!!" << std::endl;

    return 0;
}

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "net.h"
#include "benchmark.h"

#include <npu/cix_noe_standard_api.h>

class RVM_ncnn
{
public:
    void load(bool use_gpu = false)
    {
        net.opt.use_vulkan_compute = use_gpu;
        net.load_param("/home/radxa/orion-o6-rvm-cam-stream/rvm_resnet50.ncnn.param");
        net.load_model("/home/radxa/orion-o6-rvm-cam-stream/rvm_resnet50.ncnn.bin");

        r1 = ncnn::Mat(256, 256, 16);
        r2 = ncnn::Mat(128, 128, 32);
        r3 = ncnn::Mat(64, 64, 64);
        r4 = ncnn::Mat(32, 32, 128);
        r1.fill(0.0f);
        r2.fill(0.0f);
        r3.fill(0.0f);
        r4.fill(0.0f);
    }

    void run(const cv::Mat& bgr, cv::Mat& out)
    {
        ncnn::Extractor ex = net.create_extractor();

        ncnn::Mat in0 = ncnn::Mat::from_pixels(bgr.data, ncnn::Mat::PIXEL_BGR2RGB, 512, 512);

        const float mean_vals[3] = {0, 0, 0};
        const float norm_vals[3] = {1 / 255.0, 1 / 255.0, 1 / 255.0};
        in0.substract_mean_normalize(mean_vals, norm_vals);

        ex.input("in0", in0);
        ex.input("in1", r1);
        ex.input("in2", r2);
        ex.input("in3", r3);
        ex.input("in4", r4);

        ncnn::Mat fgr;
        ncnn::Mat pha;
        ex.extract("out0", fgr);
        ex.extract("out1", pha);
        ex.extract("out2", r1);
        ex.extract("out3", r2);
        ex.extract("out4", r3);
        // ex.extract("out5", r4);

        const float demean_vals[3] = {0, 0, 0};
        const float denorm_vals[3] = {255.0, 255.0, 255.0};
        fgr.substract_mean_normalize(demean_vals, denorm_vals);

        fgr.to_pixels(out.data, ncnn::Mat::PIXEL_RGB2BGR);

        // composite
        for (int y = 0; y < 512; y++)
        {
            unsigned char* p = (unsigned char*)out.data + y * 512 * 3;
            const float* ppha = (const float*)pha.data + y * 512;
            for (int x = 0; x < 512; x++)
            {
                float alpha = *ppha++;

                // 0~127 to 0~255
                p[0] = p[0] * alpha + (1 - alpha) * 155;
                p[1] = p[1] * alpha + (1 - alpha) * 255;
                p[2] = p[2] * alpha + (1 - alpha) * 120;
                p += 3;
            }
        }
    }

private:
    ncnn::Net net;

    ncnn::Mat r1;
    ncnn::Mat r2;
    ncnn::Mat r3;
    ncnn::Mat r4;
};

class RVM_noe
{
public:
    RVM_noe()
    {
        ctx = 0;
        graph_id = 0;
        job_id = 0;

        noe_init_context(&ctx);
    }

    ~RVM_noe()
    {
        noe_clean_job(ctx, job_id);
        noe_unload_graph(ctx, graph_id);
        noe_deinit_context(ctx);
    }

    void load()
    {
        noe_load_graph(ctx, "/home/radxa/orion-o6-rvm-cam-stream/rvm_resnet50.cix", &graph_id);

        noe_dynshape_param_t dynshape = {0, 0};

        job_config_npu_t job_cfg_npu;
        job_cfg_npu.partition_id = 0;
        job_cfg_npu.dbg_dispatch = 0;
        job_cfg_npu.dbg_core_id = 0;
        job_cfg_npu.fm_idxes = 0;
        job_cfg_npu.fm_idxes_cnt = 0;
        job_cfg_npu.dynshape = &dynshape;

        job_config_t job_cfg = {&job_cfg_npu};

        noe_create_job(ctx, graph_id, &job_id, &job_cfg);

        r1 = cv::Mat({16, 256, 256}, CV_8UC1);
        r2 = cv::Mat({32, 128, 128}, CV_8UC1);
        r3 = cv::Mat({64, 64, 64}, CV_8UC1);
        r4 = cv::Mat({128, 32, 32}, CV_8UC1);
        r1 = cv::Scalar(0);
        r2 = cv::Scalar(0);
        r3 = cv::Scalar(0);
        r4 = cv::Scalar(0);
    }

    void run(const cv::Mat& bgr, cv::Mat& out)
    {
        cv::Mat rgb({3, 512, 512}, CV_8UC1);

        for (int y = 0; y < 512; y++)
        {
            const unsigned char* p = (const unsigned char*)bgr.data + y * 512 * 3;
            signed char* pr = (signed char*)rgb.data + y * 512;
            signed char* pg = pr + 512 * 512;
            signed char* pb = pg + 512 * 512;
            for (int x = 0; x < 512; x++)
            {
                // 0~255 to 0~127
                *pr++ = p[0] * 127 / 255;
                *pg++ = p[1] * 127 / 255;
                *pb++ = p[2] * 127 / 255;
                p += 3;
            }
        }

        noe_load_tensor(ctx, job_id, 0, rgb.data);
        noe_load_tensor(ctx, job_id, 1, r1.data);
        noe_load_tensor(ctx, job_id, 2, r2.data);
        noe_load_tensor(ctx, job_id, 3, r3.data);
        noe_load_tensor(ctx, job_id, 4, r4.data);

        noe_job_infer_sync(ctx, job_id, 2000);

        cv::Mat fgr({3, 512, 512}, CV_8UC1);
        cv::Mat pha({1, 512, 512}, CV_8UC1);

        noe_get_tensor(ctx, job_id, NOE_TENSOR_TYPE_OUTPUT, 0, fgr.data);
        noe_get_tensor(ctx, job_id, NOE_TENSOR_TYPE_OUTPUT, 1, pha.data);
        noe_get_tensor(ctx, job_id, NOE_TENSOR_TYPE_OUTPUT, 2, r1.data);
        noe_get_tensor(ctx, job_id, NOE_TENSOR_TYPE_OUTPUT, 3, r2.data);
        noe_get_tensor(ctx, job_id, NOE_TENSOR_TYPE_OUTPUT, 4, r3.data);
        // noe_get_tensor(ctx, job_id, NOE_TENSOR_TYPE_OUTPUT, 5, r4.data);

        for (int y = 0; y < 512; y++)
        {
            unsigned char* p = (unsigned char*)out.data + y * 512 * 3;
            const unsigned char* pr = (const unsigned char*)fgr.data + y * 512;
            const unsigned char* pg = pr + 512 * 512;
            const unsigned char* pb = pg + 512 * 512;
            const unsigned char* ppha = (const unsigned char*)pha.data + y * 512;
            for (int x = 0; x < 512; x++)
            {
                float alpha = *ppha++ / 255.f;

                // 0~127 to 0~255
                p[0] = std::min((int)*pr++, 127) * 255 / 127 * alpha + (1 - alpha) * 155;
                p[1] = std::min((int)*pg++, 127) * 255 / 127 * alpha + (1 - alpha) * 255;
                p[2] = std::min((int)*pb++, 127) * 255 / 127 * alpha + (1 - alpha) * 120;
                p += 3;
            }
        }
    }

public:
    context_handler_t* ctx;
    uint64_t graph_id;
    uint64_t job_id;

    cv::Mat r1;
    cv::Mat r2;
    cv::Mat r3;
    cv::Mat r4;
};

static int draw_fps(cv::Mat& rgb)
{
    // resolve moving average
    float avg_fps = 0.f;
    {
        static double t0 = 0.f;
        static float fps_history[10] = {0.f};

        double t1 = ncnn::get_current_time();
        if (t0 == 0.f)
        {
            t0 = t1;
            return 0;
        }

        float fps = 1000.f / (t1 - t0);
        t0 = t1;

        for (int i = 9; i >= 1; i--)
        {
            fps_history[i] = fps_history[i - 1];
        }
        fps_history[0] = fps;

        if (fps_history[9] == 0.f)
        {
            return 0;
        }

        for (int i = 0; i < 10; i++)
        {
            avg_fps += fps_history[i];
        }
        avg_fps /= 10.f;
    }

    char text[32];
    sprintf(text, "FPS=%.2f", avg_fps);

    int baseLine = 0;
    cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

    int y = 0;
    int x = rgb.cols - label_size.width;

    cv::rectangle(rgb, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                    cv::Scalar(255, 255, 255), -1);

    cv::putText(rgb, text, cv::Point(x, y + label_size.height),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));

    return 0;
}

int main()
{
    RVM_ncnn rvm_cpu;
    rvm_cpu.load(false);

    RVM_ncnn rvm_gpu;
    rvm_gpu.load(true);

    RVM_noe rvm_npu;
    rvm_npu.load();

    cv::VideoCapture cap;
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 240);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 240);
    cap.open(3);

    cv::VideoWriter http;
    http.open("httpjpg", 7766);

    // open streaming url http://<server ip>:7766 in web browser

    cv::Mat bgr;
    cv::Mat bgr_512(512, 512, CV_8UC3);
    cv::Mat out(512, 512, CV_8UC3);

    while (1)
    {
        cap >> bgr;

        // rvm
        cv::resize(bgr, bgr_512, cv::Size(512, 512));

        // rvm_cpu.run(bgr_512, out);
        // rvm_gpu.run(bgr_512, out);
        rvm_npu.run(bgr_512, out);

        cv::resize(out, bgr, cv::Size(bgr.cols, bgr.rows));

        draw_fps(bgr);

        http << bgr;
    }

    return 0;
}

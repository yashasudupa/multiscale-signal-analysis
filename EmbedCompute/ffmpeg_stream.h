#ifndef FFMPEGSTREAM_H
#define FFMPEGSTREAM_H

#include <opencv2/opencv.hpp>
extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libswscale/swscale.h>
}

class FFmpegStream {
public:
    FFmpegStream(const std::string& input_url);
    cv::Mat readFrame();
    ~FFmpegStream();

private:
    AVFormatContext* format_ctx_ = nullptr;
    AVCodecContext* codec_ctx_ = nullptr;
    SwsContext* sws_ctx_ = nullptr;
    int video_stream_idx_ = -1;
};

#endif

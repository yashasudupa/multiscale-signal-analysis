#include "FFmpegStream.h"

FFmpegStream::FFmpegStream(const std::string& input_url) {
    av_register_all();
    avformat_network_init();

    if (avformat_open_input(&format_ctx_, input_url.c_str(), nullptr, nullptr) != 0) {
        throw std::runtime_error("Could not open input stream.");
    }

    if (avformat_find_stream_info(format_ctx_, nullptr) < 0) {
        throw std::runtime_error("Could not find stream information.");
    }

    for (unsigned int i = 0; i < format_ctx_->nb_streams; i++) {
        if (format_ctx_->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            video_stream_idx_ = i;
            break;
        }
    }

    if (video_stream_idx_ == -1) {
        throw std::runtime_error("Could not find video stream.");
    }

    AVCodecParameters* codec_params = format_ctx_->streams[video_stream_idx_]->codecpar;
    AVCodec* codec = avcodec_find_decoder(codec_params->codec_id);
    codec_ctx_ = avcodec_alloc_context3(codec);
    avcodec_parameters_to_context(codec_ctx_, codec_params);

    if (avcodec_open2(codec_ctx_, codec, nullptr) < 0) {
        throw std::runtime_error("Could not open codec.");
    }

    sws_ctx_ = sws_getContext(codec_ctx_->width, codec_ctx_->height, codec_ctx_->pix_fmt,
                              codec_ctx_->width, codec_ctx_->height, AV_PIX_FMT_BGR24, SWS_BILINEAR, nullptr, nullptr, nullptr);
}

cv::Mat FFmpegStream::readFrame() {
    AVPacket packet;
    av_init_packet(&packet);

    while (av_read_frame(format_ctx_, &packet) >= 0) {
        if (packet.stream_index == video_stream_idx_) {
            if (avcodec_send_packet(codec_ctx_, &packet) >= 0) {
                AVFrame* frame = av_frame_alloc();
                if (avcodec_receive_frame(codec_ctx_, frame) >= 0) {
                    cv::Mat img(codec_ctx_->height, codec_ctx_->width, CV_8UC3);
                    uint8_t* data[1] = { img.data };
                    int linesize[1] = { static_cast<int>(img.step[0]) };
                    sws_scale(sws_ctx_, frame->data, frame->linesize, 0, codec_ctx_->height, data, linesize);

                    av_frame_free(&frame);
                    av_packet_unref(&packet);
                    return img;
                }
                av_frame_free(&frame);
            }
        }
        av_packet_unref(&packet);
    }

    throw std::runtime_error("End of stream.");
}

FFmpegStream::~FFmpegStream() {
    sws_freeContext(sws_ctx_);
    avcodec_free_context(&codec_ctx_);
    avformat_close_input(&format_ctx_);
}

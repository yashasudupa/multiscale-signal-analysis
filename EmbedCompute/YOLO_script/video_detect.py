import argparse
import os
import sys
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync

import subprocess
import numpy as np
import cv2

class LoadFFmpegStream:
    def __init__(self, source, img_size=(1920, 1080)):
        self.img_size = img_size

        # Command to run ffmpeg and capture video stream
        self.ffmpeg_cmd = [
            'ffmpeg',
            '-i', 'http://192.168.1.12:8080/video',
            '-f', 'image2pipe',
            '-pix_fmt', 'bgr24',
            '-vcodec', 'rawvideo', '-'
        ]

        # Run the ffmpeg command
        self.ffmpeg_proc = subprocess.Popen(self.ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Frame size
        self.frame_size = img_size[1] * img_size[0] * 3  # width * height * channels

    def __iter__(self):
        return self

    def __next__(self):
        # Read frame from ffmpeg process
        raw_frame = self.ffmpeg_proc.stdout.read(self.frame_size)
        if len(raw_frame) != self.frame_size:
            raise StopIteration
        
        # Convert raw bytes to numpy array
        frame = np.frombuffer(raw_frame, np.uint8).reshape((self.img_size[0], self.img_size[1], 3))  # Adjust based on your frame size

        # Convert frame to BGR (OpenCV uses BGR by default)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        return frame

@torch.no_grad()
def run(weights=ROOT / 'best.pt',  # model.pt path(s)
        source='http://192.168.1.3:8080/video',  # video stream URL
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    
    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    frame_counter = 0

    try:
        while True:
            # Dataloader
            print('url : ', source)
            img_size = (1920, 1080)
            dataset = LoadFFmpegStream(source)

            # Run inference
            model.warmup(imgsz=(1, 3, img_size[1], img_size[0]))  # warmup
            dt, seen = [0.0, 0.0, 0.0], 0
            for frame in dataset:
                t1 = time_sync()
                im = torch.from_numpy(frame).to(device)
                im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
                im /= 255  # 0 - 255 to 0.0 - 1.0
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim
                t2 = time_sync()
                dt[0] += t2 - t1

                # Inference
                visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
                pred = model(im, augment=augment, visualize=visualize)
                t3 = time_sync()
                dt[1] += t3 - t2

                # NMS
                pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
                dt[2] += time_sync() - t3

                # Process predictions
                for i, det in enumerate(pred):  # per image
                    seen += 1
                    s = '%gx%g ' % im.shape[2:]  # Print string
                    gn = torch.tensor(frame.shape)[[1, 0, 1, 0]]  # Normalization gain whwh
                    imc = frame.copy() if save_crop else frame  # For save_crop
                    annotator = Annotator(frame, line_width=line_thickness, example=str(names))
                    if len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                        # Write results
                        for *xyxy, conf, cls in reversed(det):
                            if save_txt:  # Write to file
                                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                                line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                                with open(f'{txt_path}.txt', 'a') as f:
                                    f.write(('%g ' * len(line)).rstrip() % line + '\n')

                            if save_img or save_crop or view_img:  # Add bbox to image
                                c = int(cls)  # integer class
                                label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                                annotator.box_label(xyxy, label, color=colors(c, True))
                            if save_crop:
                                save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

                    # Stream results
                    im0 = annotator.result()
                    if view_img:
                        cv2.imshow(str(p), im0)
                        cv2.waitKey(1)  # 1 millisecond

                    # Save results (image with detections)
                    save_path = os.path.join(save_dir, f"frame_{frame_counter}.jpg")
                    print('Inferred and saved one image')
                    cv2.imwrite(save_path, frame)

                    frame_counter += 1
                        
            # Print results
            t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
            LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
            if save_txt or save_img:
                s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
                LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
            if update:
                strip_optimizer(weights)  # update model (to fix SourceChangeWarning)

    except KeyboardInterrupt:
        LOGGER.info('Process interrupted.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='best.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='http://100.89.98.104:8080/video', help='video stream URL')
    parser.add_argument('--data', type=str, default='data/coco128.yaml', help='dataset.yaml path')
    parser.add_argument('--imgsz', nargs='+', type=int, default=[640, 640], help='inference size (height, width)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IOU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', type=str, default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', type=str, default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', type=int, default=3, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    
    run(weights=opt.weights,
        source=opt.source,
        data=opt.data,
        imgsz=opt.imgsz,
        conf_thres=opt.conf_thres,
        iou_thres=opt.iou_thres,
        max_det=opt.max_det,
        device=opt.device,
        view_img=opt.view_img,
        save_txt=opt.save_txt,
        save_conf=opt.save_conf,
        save_crop=opt.save_crop,
        nosave=opt.nosave,
        classes=opt.classes,
        agnostic_nms=opt.agnostic_nms,
        augment=opt.augment,
        visualize=opt.visualize,
        update=opt.update,
        project=opt.project,
        name=opt.name,
        exist_ok=opt.exist_ok,
        line_thickness=opt.line_thickness,
        hide_labels=opt.hide_labels,
        hide_conf=opt.hide_conf,
        half=opt.half,
        dnn=opt.dnn)

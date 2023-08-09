import os
import sys
from pathlib import Path

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode

import cv2
import numpy as np
from lib_detection import load_model, detect_lp, im2single
from os.path import splitext, basename

wpod_net_path = 'wpod-net_update1.json'
wpod_net = load_model(wpod_net_path)

print('Searching for license plates using WPOD-NET')

img_path = "./test.jpg"
output_dir = "./Output/"
bname = splitext(basename(img_path))[0]

Ivehicle = cv2.imread(img_path)
ratio = float(max(Ivehicle.shape[:2]))/min(Ivehicle.shape[:2])
side  = int(ratio*288.)
bound_dim = min(side + (side%(2**4)),608)
Llp , LpImgs, lp_type = detect_lp(wpod_net, im2single(Ivehicle), bound_dim, lp_threshold=0.5)

if (len(LpImgs)):
    Ilp = LpImgs[0]
    name = output_dir + '%s_lp.png' % bname
    cv2.imwrite(name, Ilp*255.)

@smart_inference_mode()
def run(
        weights=ROOT / 'yolov5s.pt',  # model path or triton URL
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.5,  # confidence threshold
        iou_thres=0.5,  # NMS IOU threshold
        max_det=25,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        save_txt=True,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=2,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
        type=1,
):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt  = model.stride, model.names, model.pt

    # Image size
    if lp_type == 1:
        imgsz = (160, 640)
    elif lp_type == 2:
        imgsz = (480, 640)
    # imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Process predictions
        for i, det in enumerate(pred):  # per image

            p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            # txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            # imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescal 
                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                lines = []
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    c = int(cls)
                    label = None if hide_labels else (names[c] if hide_conf else f'{names[c]}')
                    line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                    line = label + ' ' + ('%g ' * len(line)).rstrip() % line + '\n'
                    line = line.split(' ')
                    lines.append(line)
                    # with open(f'{txt_path}.txt', 'a') as f:
                    #     f.write(line)
                    annotator.box_label(xyxy, label, color=colors(c, True))

                license_plate = ''
                # lines = np.array(lines)
                if type == 1:
                    sorted_lines = sorted(lines, key=lambda x:x[2])
                    for k in range(len(det)):
                         license_plate += sorted_lines[k][0]     
                elif type == 2:
                    sorted_lines = sorted(lines, key=lambda x:x[2])
                    sorted_lines = np.array(sorted_lines)
                    avg = sum((map(float, sorted_lines[:,3]))) / len(det)
                    # print(avg)
                    # print(sum)
                    first_row = ''
                    second_row = ''
                    for k in range(len(det)):
                        if float(sorted_lines[k][3]) < avg:
                            first_row += sorted_lines[k][0]
                        else:
                            second_row += sorted_lines[k][0]
                    license_plate = first_row + second_row

            # Save results (image with detections)        
            license_plate = license_plate + '_' + p.name 
            save_path = str(save_dir / license_plate)  # im.jpg
            print(save_path)
            im0 = annotator.result()
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)

        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # Print results
    # t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    # LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)

if __name__ == '__main__':
    # opt = parse_opt()
    # main(opt)
    run(weights='./20220919_yolo.pt', source=name, type=lp_type)
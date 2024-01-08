from face_detector import YoloDetector
import numpy as np
from PIL import Image

import argparse

def main(args):
    model = YoloDetector(
        weights_name=args.weights_name, 
        config_name=args.config_name, 
        target_size=args.target_size, 
        device=args.device,
        min_face=args.min_face
    )
    
    orgimg = np.array(Image.open(args.img_path))
    bboxes = model.predict(
        orgimg, 
        conf_thres=args.conf,
        iou_thres=args.iou,             
        return_landmarks=args.return_landmarks
    )
    print("Found {} faces.".format(len(bboxes[0])))
    
    if args.draw:
        assert args.save_dir is not None, 'Please specify a directory where to save the image.'
        from utils.plots import plot_one_box
        
        for bbox in bboxes[0]:
            plot_one_box(bbox[:4], orgimg, label=f' {bbox[4]:.2f}', color=(255,0,0))
            
        out = Image.fromarray(orgimg)
        out.save(f'{args.save_dir}/{args.img_path.split("/")[-1]}')


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Detect faces in image.')
    argparser.add_argument('img_path', help='Path to image.', type=str)
    argparser.add_argument('-w', '--weights_name', help='Name of the model weights file.', default='yolov7-face_state_dict.pth')
    argparser.add_argument('-c', '--config_name', help='Name of the model config file.', default='yolov7-face.yaml')
    argparser.add_argument('-t', '--target_size', help='Target size of the image.', default=6400, type=int)
    argparser.add_argument('-d', '--device', help='Device to run inference on.', default='cuda:0')
    argparser.add_argument('-m', '--min_face', help='Minimum face size to detect.', default=0, type=int)
    argparser.add_argument('--conf', help='Confidence threshold.', default=0.5, type=float)
    argparser.add_argument('--iou', help='IoU threshold.', default=0.5, type=float)
    argparser.add_argument('-l', '--return_landmarks', help='Return facial landmarks.', action='store_true')
    argparser.add_argument('-dr', '--draw', help='Draw bounding boxes on image.', action='store_true', default=True)
    argparser.add_argument('--save_dir', help='Directory where to save the image.', default='results')
    args = argparser.parse_args()
    
    main(args)
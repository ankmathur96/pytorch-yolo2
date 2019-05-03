import os
import sys
sys.path.append('..')
sys.path.append('../noscope/')
from noscope.video_data_utils import get_video_dataset
import argparse
import cv2
from darknet import Darknet
from utils import *
from PIL import Image
import csv

parser = argparse.ArgumentParser(description='PyTorch YoloV2 Execution on Video')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 0)')
parser.add_argument('--root-dir', default='/home/ankitmathur/noscope-videos/', type=str, metavar='PATH', help='Root directory for video data')
parser.add_argument('--video-name', default='bytes.mp4', type=str, metavar='NAME', help='Which video to execute on')
parser.add_argument('--target-label', default='person', type=str, metavar='LABEL', help='Which label for the specialized model to specialize on')
parser.add_argument('--darknet-config-file', default='cfg/yolo.cfg', type=str, metavar='PATH', help='Config file for DarkNet')
parser.add_argument('--darknet-weights-file', default='cfg/yolo.weights', type=str, metavar='PATH', help='Config file for DarkNet')
parser.add_argument('-s', '--save-pred', action='store_true', help='Set if you want to generate images with predictions') 

# returns list(list(tuple())) where each tuple is a single object prediction.
def process_batch(img_orig_height, img_orig_width, boxes):
    all_box_predictions = []
    for sample in boxes:
        predictions = []
        for box in sample:
            x1 = (box[0] - box[2]/2.0) * img_orig_width
            y1 = (box[1] - box[3]/2.0) * img_orig_height
            x2 = (box[0] + box[2]/2.0) * img_orig_width
            y2 = (box[1] + box[3]/2.0) * img_orig_height
            class_confidence = box[5]
            class_id = box[6]
            predictions.append((class_id.item(), class_confidence.item(), x1.item(), y1.item(), x2.item(), y2.item()))
        all_box_predictions.append(predictions)
    return all_box_predictions

def main(args):
    m = Darknet(args.darknet_config_file)
    m.print_network()
    m.load_weights(args.darknet_weights_file)
    print('Loaded weights')
    if m.num_classes == 20:
        names_file = 'data/voc.names'
    elif m.num_classes == 80:
        names_file = 'data/coco.names'
    else:
        print('Using default names')
        names_file = 'data/names'
    m = m.cuda()
    def input_transform(im):
        im = cv2.resize(im, (m.width, m.height))
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = im.transpose(2, 0, 1)
        return im
    class_names = load_class_names(names_file)
    train_loader, val_loader, stats = get_video_dataset(args.root_dir, args.video_name, batch_size=args.batch_size, input_transform=input_transform, use_dummy_labels=True)
    c = 0
    labels_path = os.path.join(args.root_dir, 'labels', args.video_name + '.csv')
    with open(labels_path, 'w') as csv_file:
        pred_writer = csv.writer(csv_file, delimiter=',')
        for i, orig_img in train_loader:
            batch_boxes = do_detect(m, i, 0.5, 0.4, use_cuda=True)
            predictions = process_batch(stats['height'], stats['width'], batch_boxes)
            if args.save_pred:
                orig_img = orig_img.numpy()
                save_path = args.video_name + '_ims'
                if not os.path.exists(save_path):
                    print('Created save directory for prediction images: {}'.format(save_path))
                    os.makedirs(save_path)
            for i, sample_preds in enumerate(predictions):
                for pred in sample_preds:
                    class_id, class_confidence, x1, y1, x2, y2 = pred
                    class_name = class_names[class_id]
                    pred_writer.writerow([c] + [class_name, class_id, class_confidence, x1, y1, x2, y2])
                if args.save_pred:
                    pil_img = Image.fromarray(orig_img[i])
                    zfill_number = '{0:09d}'.format(c)
                    plot_boxes(pil_img, batch_boxes[i], savename=os.path.join(save_path, 'predictions_' + zfill_number + '.jpg'), class_names=class_names)
                c += 1
            print('Inference: c = {}'.format(c))



if __name__ == '__main__':
    args = parser.parse_args()
    main(args)


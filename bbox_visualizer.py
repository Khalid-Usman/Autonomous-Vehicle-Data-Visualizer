import os
import sys
import cv2
import argparse
import numpy as np
import pandas as pd
from typing import List, Dict
from dataset.bin.selection_pipeline import read_json
from dataset.data_filtration.utils.label_convertor import convert_labels_to_frame_level
from visualization.data_visualization.bbox_visualizer.utils.draw_boxes_utils import draw_bounding_boxes, \
    put_text_on_image


def generate_color_dict(d: Dict, attributes: List) -> Dict:
    """
    This function will generate a dictionary for each attribute and assign a different color to a bounding box

    :param d: existing color dictionary
    :param attributes: list of unique keys
    :return d: updated color dictionary
    """
    for k in attributes:
        if k not in d.keys():
            d[k] = list(np.random.random(size=3) * 256)
    return d


def check_file_exists(path: str, name: str):
    """
    This function will check if file path do not exist then it raise error

    :param path: path of file
    :param name: name of file
    """
    if not os.path.exists(path):
        raise FileNotFoundError("{name} does not exist.".format(name=name))


def annotate_images(pdp_path: str, target_path: str, result_path: str, images: List, annotations: Dict):
    """
    This function will load images and write back after drawing bounding boxes on it

    :param pdp_path: path of pdp which contain folders e.g. images
    :param target_path: name of the target folder
    :param result_path: path of result.csv file where frame name and corresponding scores are dumped
    :param images: list of frames_name
    :param annotations: list of bounding boxes for each frame
    """
    check_file_exists(result_path, "csv file")
    max_columns = max(open(result_path, 'r'), key=lambda x: x.count(' ')).count(' ')
    df = pd.read_csv(result_path, header=None, sep=' ', usecols=range(0, max_columns))
    color_dict = {}
    for name in images:
        if (df.iloc[:, 2] == name[:-4]).any():
            label = df[df.iloc[:, 2] == name[:-4]].dropna(axis='columns').values.flatten().tolist()
            frame_boxes = annotations[label[0]]['boxes']
            frame_attributes = [d['name'] for d in annotations[label[0]]['attributes']]
            color_dict = generate_color_dict(color_dict, list(set(frame_attributes)))
            check_file_exists(os.path.join(pdp_path, 'image_color', name), 'frame')
            source_path, target_dir = os.path.join(pdp_path, 'image_color', name), os.path.join(pdp_path, target_path)
            img = draw_bounding_boxes(source_path, frame_boxes, frame_attributes, color_dict)
            img = put_text_on_image(img, label[3:])
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)
            if os.path.exists(target_dir):
                cv2.imwrite(os.path.join(target_dir, name), img)


def main(pdp_path: str, target_path: str, file_path: str, result_path: str):
    """
    This function will read all the given frames from the pdp folder and generate a key-value pair where name of frame
    is stored against the unique number of frame

    :param pdp_path: path of pdp which contain folders e.g. images
    :param target_path: name of the target folder
    :param file_path: path of JSON file, where the annotation of each frame exists
    :param result_path: path of result.csv file where frame name and corresponding scores are dumped
    """
    check_file_exists(os.path.join(pdp_path, file_path), 'annotation file')
    labels = read_json(os.path.join(pdp_path, file_path))
    annotations, attribute_list = convert_labels_to_frame_level(labels)
    check_file_exists(os.path.join(pdp_path, 'image_color'), 'images folder')
    images = os.listdir(os.path.join(pdp_path, 'image_color'))
    annotate_images(pdp_path, target_path, result_path, images, annotations)


def parse_args(sys_args: List[str]) -> argparse.Namespace:
    """
    This function parses the command line arguments.

    :param sys_args: Command line arguments
    :returns argparse namespace
    """
    parser = argparse.ArgumentParser(description='Execute bbox_labeler to show bounding boxes and text on image')
    parser.add_argument('--pdp', '-s', required=True, type=str, help='path of pdp that contain color images in its sub'
                                                                     ' directory')
    parser.add_argument('--target', '-t', required=True, type=str, help='name of the folder where you want to save '
                                                                        'output images')
    parser.add_argument('--annotation_file', '-a', required=True, type=str, help='name of a JSON file that contains '
                                                                                 'annotations of bounding boxes')
    parser.add_argument('--result_file', '-r', required=True, type=str, help='path of csv file that is generated using'
                                                                             ' data_selection pipeline')
    return parser.parse_args(sys_args)


if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    main(args.pdp, args.target, args.annotation_file, args.result_file)

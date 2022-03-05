import cv2
from typing import List, Tuple, Dict


def draw_bounding_boxes(image_path: str, boxes: List, attributes: List, color_dict: Dict):
    """Draw bounding boxes on an image.
    :param image_path: path of the underlying image
    :param boxes: a list of bounding boxes on a frame
    :param attributes: list of unique objects
    :param color_dict: dictionary with attributes as keys and colors as values

    :return img: image with bounding boxed in the given color
    """
    img = cv2.imread(image_path)
    for i in range(len(boxes)):
        x_min, y_min, x_max, y_max = int(boxes[i][0]), int(boxes[i][1]), int(boxes[i][2]), int(boxes[i][3])
        imgHeight, imgWidth, _ = img.shape
        thick = int((imgHeight + imgWidth) // 900)
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color_dict[attributes[i]], thickness=thick)
    return img


def put_text_on_image(img, label: List, font_name=cv2.FONT_HERSHEY_SIMPLEX, font_scale: float = 1.5,
                      text_color: Tuple[int, int, int] = (0, 0, 255), text_thickness: int = 2, text_Y: int = 60):
    """
    :param img: image on which we want to put text
    :param label: put a text on a frame
    :param font_name: name of font
    :param font_scale: size of font
    :param text_color: size of font
    :param text_thickness: thickness of text
    :param text_Y: top space between image and text

    :return img: image with given text in the given color
    """
    text = "score: " + ', '.join(str(v) for v in label)
    text_size = cv2.getTextSize(text, font_name, font_scale, text_thickness)[0]
    textX, textY = round((img.shape[1] - text_size[0]) / 2), text_Y
    cv2.putText(img, text, (textX, textY), font_name, font_scale, text_color, text_thickness)
    return img

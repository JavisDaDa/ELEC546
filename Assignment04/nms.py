import numpy as np
import cv2


def nms(bboxs, thresh):

    # get all parameters
    x1, y1, x2, y2, scores = [bboxs[:, i] for i in range(len(bboxs[0]))]

    # calculate all areas of boxed
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    # sort boxes according to their class score
    sorted_index = scores.argsort()[::-1]

    # result list
    result = []

    while sorted_index.size > 0:
        # get the box with largest score
        max_box = bboxs[sorted_index[0]]

        # add it to our result
        result.append(max_box)

        # calculate intersection coordinates
        xx1 = np.maximum(max_box[0], x1[sorted_index[1:]])
        yy1 = np.maximum(max_box[1], y1[sorted_index[1:]])
        xx2 = np.minimum(max_box[2], x2[sorted_index[1:]])
        yy2 = np.minimum(max_box[3], y2[sorted_index[1:]])

        # calculate intersection area
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        intersection = w * h

        # calculate ious
        ious = intersection / (areas[sorted_index[0]] + areas[sorted_index[1:]] - intersection)

        # retain all the boxes whose ious are less than the threshold
        sorted_index = sorted_index[1:][ious <= thresh]

    return result


def draw_bbox(bboxs, pic_name):
    pic = np.zeros((850, 850), np.uint8)
    for bbox in bboxs:
        x1, y1, x2, y2 = map(int, bbox[:-1])
        pic = cv2.rectangle(pic, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.imshow(pic_name,pic)
    cv2.waitKey(0)


if __name__ == "__main__":
    bboxs = np.array([
        [720, 690, 820, 800, 0.5],
        [204, 102, 358, 250, 0.5],
        [257, 118, 380, 250, 0.8],
        [700, 700, 800, 800, 0.4],
        [280, 135, 400, 250, 0.7],
        [255, 118, 360, 235, 0.7]])
    thresh = 0.3
    draw_bbox(bboxs, "Before_NMS")
    result = nms(bboxs, thresh)
    draw_bbox(result, "After_NMS")
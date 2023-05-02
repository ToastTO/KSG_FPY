import numpy as np

def get_det_boxes(result,classes,conf_thres = 0.25,iou_thres = 0.8,max_det=10):
    pred = non_max_suppression(result, conf_thres, iou_thres, None, False, max_det=max_det)
    processed_result = []
    for i, det in enumerate(pred): 
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4][:, [0, 2]] 
            det[:, :4][:, [1, 3]] 
   
            # Write results
            for *xyxy, conf, cls in reversed(det):    
                # print(xyxy[0],xyxy[1],xyxy[2],xyxy[3])  
                processed_result.append({
                    'left': int(xyxy[0]),'top': int(xyxy[1]), 
                    'right': int(xyxy[2]),'bottom': int(xyxy[3]), 
                    'score': conf,'id':classes[int(cls)]})
    return processed_result

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def box_area(box):
    # box = xyxy(4,n)
    return (box[2] - box[0]) * (box[3] - box[1])


def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (ndarray[N, 4])
        box2 (ndarray[M, 4])
    Returns:
        iou (ndarray[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    (a1, a2), (b1, b2) = np.split(box1, 2, axis=1), np.split(box2, 2, axis=1)
    inter = np.clip(np.minimum(a2, b2) - np.maximum(a1, b1), 0, None).prod(1)

    # IoU = inter / (area1 + area2 - inter)
    area1 = box_area(box1.T)
    area2 = box_area(box2.T)
    return inter / (area1[:, None] + area2 - inter)

def non_max_suppression(prediction,
                        conf_thres=0.25,
                        iou_thres=0.45,
                        classes=None,
                        agnostic=False,
                        multi_label=False,
                        labels=(),
                        max_det=300):
    """Non-Maximum Suppression (NMS) on inference results to reject overlapping bounding boxes

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """
    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    prediction = np.array(prediction)
    bs = prediction.shape[0]  # batch size
    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    # min_wh = 2  # (pixels) minimum box width and height
    max_wh = 7680  # (pixels) maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 0.3 + 0.03 * bs  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    # make a list which contains #bs arrays 
    # if bs = 2, output = [ array1 , array2 ]
    # and each array has shape(0,6) whish means empty! 
    # thats why ouput is a list!
    output = [np.zeros((0, 6))] * bs

    # loop over bs xi=bs index, x=image
    for xi, x in enumerate(prediction):  # image index, image inference

        # filter out all the 
        x = x[xc[xi]]  # confidence

        # If none remain process next image
        if not x.shape[0]:
        #   print(x.shape)
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf
        # print(x.shape)

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        #Best class seleted only
        conf = np.max(x[:, 5:], axis=1, keepdims=True)
        j = np.argmax(x[:, 5:], axis=1).reshape(-1, 1)
        x = np.concatenate((box, conf, j.astype(float)), 1)[conf.ravel() > conf_thres]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort()[::-1][:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = np.array(py_cpu_nms(boxes, scores, iou_thres))  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = (weights @ x[:, :4]).astypr(float) / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
    return output

def py_cpu_nms(dets, scores,thresh):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep

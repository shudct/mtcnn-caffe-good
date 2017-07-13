#!/usr/bin/env python
# -*- coding: utf-8 -*-

import _init_paths
import caffe
import cv2
import numpy as np

import time


def preprocess_cvimg(cv_img):
    # BGR->RGB
    img = cv_img[:, :, (2, 1, 0)]
    # Normalize
    img = (img - 127.5) * 0.0078125  # [0,255] -> [-1,1]

    return img


def adjust_input(in_data):
    """
        adjust the input from (h, w, c) to ( 1, c, h, w) for network input

    Parameters:
    ----------
        in_data: numpy array of shape (h, w, c)
            input data
    Returns:
    -------
        out_data: numpy array of shape (1, c, h, w)
            reshaped array
    """
    if in_data.dtype is not np.dtype('float32'):
        out_data = in_data.astype(np.float32)
    else:
        out_data = in_data

#    print out_data.shape
#    cv2.imshow('resized', out_data)
#    cv2.waitKey(0)

#    out_data = (out_data - 127.5)*0.0078125

    out_data = np.expand_dims(out_data, 0)
    out_data = np.swapaxes(out_data, 1, 3)

    return out_data


def bbox_reg(bbox, reg):
    reg = reg.T

    # calibrate bouding boxes
    if reg.shape[1] == 1:
        #print("reshape of reg")
        pass  # reshape of reg

    w = bbox[:, 2] - bbox[:, 0] + 1
    h = bbox[:, 3] - bbox[:, 1] + 1

    bb0 = bbox[:, 0] + reg[:, 0] * w
    bb1 = bbox[:, 1] + reg[:, 1] * h
    bb2 = bbox[:, 2] + reg[:, 2] * w
    bb3 = bbox[:, 3] + reg[:, 3] * h

    bbox[:, 0:4] = np.array([bb0, bb1, bb2, bb3]).T
    # #print "bb", bbox
    return bbox


def pad(boxesA, w, h):
    boxes = boxesA.copy()  # shit, value parameter!!!

    # #print 'boxes', boxes
    # #print 'w,h', w, h

    tmph = boxes[:, 3] - boxes[:, 1] + 1
    tmpw = boxes[:, 2] - boxes[:, 0] + 1
    n_boxes = boxes.shape[0]

    tmph = tmph.astype(np.int32)
    tmpw = tmpw.astype(np.int32)

    # #print 'tmph', tmph
    # #print 'tmpw', tmpw

    dx = np.ones(n_boxes, np.int32)
    dy = np.ones(n_boxes, np.int32)
    edx = tmpw
    edy = tmph

    x = (boxes[:, 0:1][:, 0]).astype(np.int32)
    y = (boxes[:, 1:2][:, 0]).astype(np.int32)
    ex = (boxes[:, 2:3][:, 0]).astype(np.int32)
    ey = (boxes[:, 3:4][:, 0]).astype(np.int32)

    tmp = np.where(ex > w)[0]
    if tmp.shape[0] != 0:
        edx[tmp] = -ex[tmp] + w - 1 + tmpw[tmp]
        ex[tmp] = w - 1

    tmp = np.where(ey > h)[0]
    if tmp.shape[0] != 0:
        edy[tmp] = -ey[tmp] + h - 1 + tmph[tmp]
        ey[tmp] = h - 1

    tmp = np.where(x < 1)[0]
    if tmp.shape[0] != 0:
        dx[tmp] = 2 - x[tmp]
        x[tmp] = np.ones_like(x[tmp])

    tmp = np.where(y < 1)[0]
    if tmp.shape[0] != 0:
        dy[tmp] = 2 - y[tmp]
        y[tmp] = np.ones_like(y[tmp])

    # for python index from 0, while matlab from 1
    dy = np.maximum(0, dy - 1)
    dx = np.maximum(0, dx - 1)
    y = np.maximum(0, y - 1)
    x = np.maximum(0, x - 1)
    edy = np.maximum(0, edy - 1)
    edx = np.maximum(0, edx - 1)
    ey = np.maximum(0, ey - 1)
    ex = np.maximum(0, ex - 1)

    # #print "dy"  ,dy
    # #print "dx"  ,dx
    # #print "y "  ,y
    # #print "x "  ,x
    # #print "edy" ,edy
    # #print "edx" ,edx
    # #print "ey"  ,ey
    # #print "ex"  ,ex

    # #print 'boxes', boxes
    return [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph]


def convert_to_squares(bboxA):
    # convert bboxA to square
    w = bboxA[:, 2] - bboxA[:, 0]
    h = bboxA[:, 3] - bboxA[:, 1]
    l = np.maximum(w, h).T

    # #print 'bboxA', bboxA
    # #print 'w', w
    # #print 'h', h
    # #print 'l', l
    bboxA[:, 0] = bboxA[:, 0] + w * 0.5 - l * 0.5
    bboxA[:, 1] = bboxA[:, 1] + h * 0.5 - l * 0.5
    bboxA[:, 2:4] = bboxA[:, 0:2] + np.repeat([l], 2, axis=0).T
    return bboxA


def nms(boxes, threshold, type):
    """nms
    :boxes: [:,0:5]
    :threshold: 0.5 like
    :type: 'Min' or others
    :returns: TODO
    """
    if boxes.shape[0] == 0:
        return np.array([])
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    s = boxes[:, 4]
    area = np.multiply(x2 - x1 + 1, y2 - y1 + 1)
    I = np.array(s.argsort())  # read s using I

    pick = []
    while len(I) > 0:
        xx1 = np.maximum(x1[I[-1]], x1[I[0:-1]])
        yy1 = np.maximum(y1[I[-1]], y1[I[0:-1]])
        xx2 = np.minimum(x2[I[-1]], x2[I[0:-1]])
        yy2 = np.minimum(y2[I[-1]], y2[I[0:-1]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        if type == 'Min':
            o = inter / np.minimum(area[I[-1]], area[I[0:-1]])
        else:
            o = inter / (area[I[-1]] + area[I[0:-1]] - inter)
        pick.append(I[-1])
        I = I[np.where(o <= threshold)[0]]
    return pick


def generate_bboxes(scores_map, reg, scale, t):
    stride = 2
    cellsize = 12

    scores_map = scores_map.T

    dx1 = reg[0, :, :].T
    dy1 = reg[1, :, :].T
    dx2 = reg[2, :, :].T
    dy2 = reg[3, :, :].T

    (x, y) = np.where(scores_map >= t)

    if len(x) < 1:
        return None

    yy = y
    xx = x

    scores = scores_map[x, y]
    reg = np.array([dx1[x, y], dy1[x, y], dx2[x, y], dy2[x, y]])

    if reg.shape[0] == 0:
        pass

    bbox = np.array([yy, xx]).T

    # matlab index from 1, so with "bbox-1"
    bb1 = np.fix((stride * (bbox) + 1) / scale).T
    bb2 = np.fix((stride * (bbox) + cellsize - 1 + 1) /
                 scale).T  # while python don't have to
    scores = np.array([scores])

    bbox_out = np.concatenate((bb1, bb2, scores, reg), axis=0)

    # #print '(x,y)',x,y
    # #print 'scores', scores
    # #print 'reg', reg

    return bbox_out.T


def align_face(aligner, cv_img,  face_rects):
    RNet, ONet, LNet = aligner

    if not face_rects:
        return ([], [])

    n_boxes = len(face_rects)

    img = cv_img.copy()
    img = preprocess_cvimg(img)


#    factor_count = 0
#    total_boxes = np.zeros((0, 9), np.float)
#    points = []
#
    h = img.shape[0]
    w = img.shape[1]
#    minl = min(h, w)
#
#    m = 12.0 / minsize
#    minl = minl * m
#
#    # create scale pyramid
#    scales = []
#    while minl >= 12:
#        scales.append(m * pow(factor, factor_count))
#        minl *= factor
#        factor_count += 1
#
#    ###############
#    # First stage
#    ###############
#    # t1 = time.clock()
#
#    # 1.1 run PNet
#    for scale in scales:
#        hs = int(np.ceil(h * scale))
#        ws = int(np.ceil(w * scale))
#
#        im_data = cv2.resize(img, (ws, hs))  # default is bilinear
#        im_data = np.swapaxes(im_data, 0, 2)
#
#        PNet.blobs['data'].reshape(1, 3, ws, hs)
#        PNet.blobs['data'].data[...] = im_data
#        out = PNet.forward()
#
#        boxes = generate_bboxes(
#            out['prob1'][0, 1, :, :], out['conv4-2'][0], scale, threshold[0])
#
#        if boxes is None:
#            continue
#
#        if boxes.shape[0] > 0:
#            pick = nms(boxes, 0.5, 'Union')
#
#            if len(pick) > 0:
#                boxes = boxes[pick, :]
#            else:
#                continue
#
#            total_boxes = np.concatenate((total_boxes, boxes), axis=0)
#
#    # t2 = time.clock()
#    #print("-->PNet cost %f seconds, processed %d pyramid scales" % ((t2-t1), len(scales)) )
#
#    n_boxes = total_boxes.shape[0]
#    # #print("-->PNet outputs #total_boxes = %d" % n_boxes)
#
#    if n_boxes < 1:
#        return ([], [])
#
#    # 1.2 run NMS
#    # t1 = time.clock()
#
#    pick = nms(total_boxes, 0.7, 'Union')
#
#    # t2 = time.clock()
#    #print("-->First NMS cost %f seconds, processed %d boxes" % ((t2-t1), n_boxes) )
#
#    n_boxes = len(pick)
#    #print("-->First NMS outputs %d boxes" % n_boxes )
#
#    total_boxes = total_boxes[pick, :]
#
#    # revise and convert to square
#    regh = total_boxes[:, 3] - total_boxes[:, 1]
#    regw = total_boxes[:, 2] - total_boxes[:, 0]
#
#    t1 = total_boxes[:, 0] + total_boxes[:, 5] * regw
#    t2 = total_boxes[:, 1] + total_boxes[:, 6] * regh
#    t3 = total_boxes[:, 2] + total_boxes[:, 7] * regw
#    t4 = total_boxes[:, 3] + total_boxes[:, 8] * regh
#    t5 = total_boxes[:, 4]

#    total_boxes = np.array([t1, t2, t3, t4, t5]).T

    total_boxes = np.array(face_rects)

    # convert list of 4 pts (x1,y1,x2,y2,x3,y3,x4,y4) into list of 2 pts
    # (x1,y1,x3,y3)
    if total_boxes.ndim == 3:
        #        t_shape = total_boxes.shape
        #        total_boxes = total_boxes.reshape((-1, t_shape[1]*t_shape[2]))
        #        total_boxes = total_boxes[:, (0,1,4,5)]
        total_boxes = total_boxes[:, (0, 2), :]
        total_boxes = total_boxes.reshape((-1, 4))

    print total_boxes

    if RNet:
        total_boxes = convert_to_squares(total_boxes)  # convert box to square

        total_boxes[:, 0:4] = np.fix(total_boxes[:, 0:4])
        [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = pad(total_boxes, w, h)

        ###############
        # Second stage
        ###############
        # t1 = time.clock()

        # 2.1 construct input for RNet
        tmp_img = np.zeros((n_boxes, 24, 24, 3))  # (24, 24, 3, n_boxes)
        for k in range(n_boxes):
            tmp = np.zeros((int(tmph[k]), int(tmpw[k]), 3))

            tmp[dy[k]:edy[k] + 1, dx[k]:edx[k] +
                1] = img[y[k]:ey[k] + 1, x[k]:ex[k] + 1]

            tmp_img[k, :, :, :] = cv2.resize(tmp, (24, 24))

        # 2.2 run RNet
        tmp_img = np.swapaxes(tmp_img, 1, 3)

        RNet.blobs['data'].reshape(n_boxes, 3, 24, 24)
        RNet.blobs['data'].data[...] = tmp_img
        out = RNet.forward()

    #    scores = out['prob1'][:, 1]
        reg_factors = out['conv5-2'].T

    #    scores = out['prob1'][:, 1]
    #    pass_t = np.where(scores > threshold[1])[0]

        # t2 = time.clock()
        #print("-->RNet cost %f seconds, processed %d boxes, avg time: %f seconds" % ((t2-t1), n_boxes, (t2-t1)/n_boxes) )

    #    n_boxes = pass_t.shape[0]
    #    # #print("-->RNet outputs #total_boxes = %d" % n_boxes)
    #
    #    if n_boxes < 1:
    #        return ([], [])
    #
    #    scores = np.array([scores[pass_t]]).T
    #    total_boxes = np.concatenate((total_boxes[pass_t, 0:4], scores), axis=1)
    #    reg_factors = out['conv5-2'][pass_t, :].T

    #    # 2.3 NMS
    #    # t1 = time.clock()
    #    pick = nms(total_boxes, 0.7, 'Union')
    #    # t2 = time.clock()
    #
    #    #print("-->Second NMS cost %f seconds, processed %d boxes" % ((t2-t1), n_boxes) )
    #
    #    n_boxes = len(pick)
    #    #print("-->Second NMS outputs %d boxes" % n_boxes )
    #
    #    if n_boxes < 1:
    #        return ([], [])
    #
    #    total_boxes = total_boxes[pick, :]
    #    total_boxes = bbox_reg(total_boxes, reg_factors[:, pick])
    #    total_boxes = convert_to_squares(total_boxes)
    #

        total_boxes = bbox_reg(total_boxes, reg_factors)

    ###############
    # Third stage
    ###############
    total_boxes = convert_to_squares(total_boxes)

    # 3.1 construct input for ONet

    total_boxes = np.fix(total_boxes)
    [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = pad(total_boxes, w, h)

    tmp_img = np.zeros((n_boxes, 48, 48, 3))
    for k in range(n_boxes):
        tmp = np.zeros((tmph[k], tmpw[k], 3))
        tmp[dy[k]:edy[k] + 1, dx[k]:edx[k] +
            1] = img[y[k]:ey[k] + 1, x[k]:ex[k] + 1]
        tmp_img[k, :, :, :] = cv2.resize(tmp, (48, 48))
#            tmp_img = (tmp_img - 127.5) * 0.0078125  # [0,255] -> [-1,1]

    # 3.2 run ONet
    # t1 = time.clock()

    tmp_img = np.swapaxes(tmp_img, 1, 3)
    ONet.blobs['data'].reshape(n_boxes, 3, 48, 48)
    ONet.blobs['data'].data[...] = tmp_img
    out = ONet.forward()

    scores = out['prob1'][:, 1]
    points = out['conv6-3']
#    pass_t = np.where(scores > threshold[2])[0]
#
    # t2 = time.clock()
    #print("-->ONet cost %f seconds, processed %d boxes, avg time: %f seconds" % ((t2-t1), n_boxes, (t2-t1)/n_boxes ))
#
#    n_boxes = pass_t.shape[0]
#    # #print("-->ONet outputs #total_boxes = %d" % n_boxes)
#
#    if n_boxes < 1:
#        return ([], [])
#
#    points = points[pass_t, :]
#    scores = np.array([scores[pass_t]]).T
#    total_boxes = np.concatenate(
#        (total_boxes[pass_t, 0:4], scores), axis=1)
# print "[9]:", total_boxes.shape[0]
#
#    reg_factors = out['conv6-2'][pass_t, :].T
#    w = total_boxes[:, 3] - total_boxes[:, 1] + 1
#    h = total_boxes[:, 2] - total_boxes[:, 0] + 1
#
#    points[:, 0:5] = np.tile(
#        w, (5, 1)).T * points[:, 0:5] + np.tile(total_boxes[:, 0], (5, 1)).T - 1
#    points[:, 5:10] = np.tile(
#        h, (5, 1)).T * points[:, 5:10] + np.tile(total_boxes[:, 1], (5, 1)).T - 1
#
#    total_boxes = bbox_reg(total_boxes, reg_factors[:, :])
#
#    # t1 = time.clock()
#    pick = nms(total_boxes, 0.7, 'Min')
#    # t2 = time.clock()
#
#    #print("-->Third NMS cost %f seconds, processed %d boxes" % ((t2-t1), n_boxes) )
#
#    n_boxes = len(pick)
#    #print("-->Third NMS outputs %d boxes" % n_boxes )
#
#    if n_boxes < 1:
#        return ([], [])
#
#    total_boxes = total_boxes[pick, :]
#    points = points[pick, :]

    scores = scores.reshape((-1, 1))
#    print scores
    total_boxes = np.concatenate(
        (total_boxes[:, 0:4], scores), axis=1)
#            #print "[9]:", total_boxes.shape[0]

    reg_factors = out['conv6-2'].T
    boxes_w = total_boxes[:, 3] - total_boxes[:, 1] + 1
    boxes_h = total_boxes[:, 2] - total_boxes[:, 0] + 1

    points[:, 0:5] = np.tile(boxes_w, (5, 1)).T * points[:, 0:5] \
                     + np.tile(total_boxes[:, 0], (5, 1)).T - 1
    points[:, 5:10] = np.tile(boxes_h, (5, 1)).T * points[:, 5:10] \
                      + np.tile(total_boxes[:, 1], (5, 1)).T - 1


    total_boxes = bbox_reg(total_boxes, reg_factors)

    ###############
    # Extended stage
    ###############
    if LNet is not None:
        # 4.1 construct input for LNet
#        total_boxes = np.fix(total_boxes)
        patchw = np.maximum(total_boxes[:, 2]-total_boxes[:, 0]+1,
                            total_boxes[:, 3]-total_boxes[:, 1]+1)

        patchw = np.round(patchw*0.25)

        # make it even
        patchw[np.where(np.mod(patchw,2) == 1)] += 1

        pointx = np.zeros((n_boxes, 5))
        pointy = np.zeros((n_boxes, 5))

        tmp_img = np.zeros((n_boxes, 15, 24, 24), dtype=np.float32)
        for i in range(5):
            x, y = points[:, i], points[:, i+5]
            x, y = np.round(x-0.5*patchw), np.round(y-0.5*patchw)
            [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = pad(
                            np.vstack([x, y, x+patchw-1, y+patchw-1]).T,
                            w, h)
            for j in range(n_boxes):
                tmpim = np.zeros((tmpw[j], tmpw[j], 3), dtype=np.float32)
                tmpim[dy[j]:edy[j]+1, dx[j]:edx[j]+1, :] = img[y[j]:ey[j]+1, x[j]:ex[j]+1, :]
                tmp_img[j, i*3:i*3+3, :, :] = adjust_input(cv2.resize(tmpim, (24, 24)))

        # 4.2 run LNet
        # t1 = time.clock()

        LNet.blobs['data'].reshape(n_boxes, 15, 24, 24)
        LNet.blobs['data'].data[...] = tmp_img
        out = LNet.forward()

#        print('--->LNet output: \n{}'.format(out))

        for k in range(5):
            # do not make a large movement
            layer_name = 'fc5_' + str(k+1)
            tmp_index = np.where(np.abs(out[layer_name]-0.5) > 0.35)
            out[layer_name][tmp_index[0]] = 0.5

            pointx[:, k] = np.round(points[:, k] - 0.5*patchw) \
                          + out[layer_name][:, 0]*patchw
            pointy[:, k] = np.round(points[:, k+5] - 0.5*patchw) \
                          + out[layer_name][:, 1]*patchw

#        print('--->LNet output: \n{}'.format(out))

        points = np.hstack([pointx, pointy])

        # t2 = time.clock()
        #print("-->LNet cost %f seconds, processed %d boxes, avg time: %f seconds" % ((t2-t1), n_boxes, (t2-t1)/n_boxes ))

    return total_boxes.tolist(), points.tolist()


def get_aligner(caffe_model_path, use_more_stage=False):
    #    minsize = 20
    #     caffe_model_path = "/home/duino/iactive/mtcnn/model"
    #    threshold = [0.6, 0.7, 0.7]
    #    factor = 0.709
    # caffe.set_mode_cpu()
    caffe.set_mode_gpu()
#    PNet = caffe.Net(caffe_model_path + "/det1.prototxt",
#                     caffe_model_path + "/det1.caffemodel", caffe.TEST)
    if use_more_stage:
        RNet = caffe.Net(caffe_model_path + "/det2.prototxt",
                         caffe_model_path + "/det2.caffemodel", caffe.TEST)
    else:
        RNet = None

    ONet = caffe.Net(caffe_model_path + "/det3.prototxt",
                     caffe_model_path + "/det3.caffemodel", caffe.TEST)

    LNet = caffe.Net(caffe_model_path + "/det4.prototxt",
                     caffe_model_path + "/det4.caffemodel", caffe.TEST)

#    return (PNet, RNet, ONet)
    return (RNet, ONet, LNet)
#    return (RNet, ONet, None)



def draw_faces(img, bboxes, points=None):
    if len(bboxes) < 1:
        pass

    for i in range(len(bboxes)):
        cv2.rectangle(img, (int(bboxes[i][0]), int(bboxes[i][1])), (int(
            bboxes[i][2]), int(bboxes[i][3])), (0, 255, 0), 1)

        if points is not None:
            for j in range(5):
                cv2.circle(img, (int(points[i][j]), int(
                    points[i][j + 5])), 2, (0, 0, 255), -1)


class MtcnnAligner:
    def __init__(self, caffe_model_path, use_more_stage=False):
        self.detector = get_aligner(caffe_model_path, use_more_stage)

    def align_face(self, img, face_rects):
        if isinstance(img, str):
            img = cv2.imread(img)

        return align_face(self.detector, img, face_rects)


if __name__ == "__main__":
    import os
    import os.path as osp
    import json

    show_img = True

    caffe_model_path = '../model'
    save_dir = './fa_rlt'
    save_json = 'mtcnn_align_test_rlt.json'

    if not osp.exists(save_dir):
        os.makedirs(save_dir)

    img_path = '../test_imgs/Marilyn_Monroe_0002.jpg'
    face_rects = [
        [
            [
                91,
                57
            ],
            [
                173,
                57
            ],
            [
                173,
                180
            ],
            [
                91,
                180
            ]
        ]
    ]


    fp_rlt = open(osp.join(save_dir, save_json), 'w')
    results = []

    img = cv2.imread(img_path)

    aligner = MtcnnAligner(caffe_model_path, False)

    t1 = time.clock()
    bboxes, points = aligner.align_face(img, face_rects)
    t2 = time.clock()

    rlt = {}
    rlt["filename"] = img_path
    rlt["faces"] = []
    rlt['face_count'] = 0

    n_boxes = len(face_rects)
    print("-->Alignment cost %f seconds, processed %d face rects, avg time: %f seconds" %
          ((t2 - t1), n_boxes, (t2 - t1) / n_boxes))

    if bboxes is not None and len(bboxes) > 0:
        for (box, pts) in zip(bboxes, points):
            #                box = box.tolist()
            #                pts = pts.tolist()
            tmp = {'rect': box[0:4],
                   'score': box[4],
                   'pts': pts
                   }
            rlt['faces'].append(tmp)

    rlt['face_count'] = len(bboxes)

    rlt['message'] = 'success'
    results.append(rlt)

    json.dump(results, fp_rlt, indent=4)
    fp_rlt.close()

    draw_faces(img, bboxes, points)
    base_name = osp.basename(img_path)
    name, ext = osp.splitext(base_name)
    ext = '.png'

    save_name = osp.join(save_dir, name + ext)
    cv2.imwrite(save_name, img)

    if show_img:
        cv2.imshow('img', img)

        cv2.waitKey(0)

        cv2.destroyAllWindows()

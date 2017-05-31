#!/usr/bin/env python
# -*- coding: utf-8 -*-

import _init_paths
import caffe
import cv2
import numpy as np

import os
import os.path as osp

import time


def preprocess_cvimg(cv_img):
    # BGR->RGB
    img = cv_img[:, :, (2, 1, 0)]
    # Normalize
    img = (img - 127.5) * 0.0078125  # [0,255] -> [-1,1]

    return img


def bbox_reg(bbox, reg):
    reg = reg.T

    # calibrate bouding boxes
    if reg.shape[1] == 1:
        print("reshape of reg")
        pass  # reshape of reg

    w = bbox[:, 2] - bbox[:, 0] + 1
    h = bbox[:, 3] - bbox[:, 1] + 1

    bb0 = bbox[:, 0] + reg[:, 0] * w
    bb1 = bbox[:, 1] + reg[:, 1] * h
    bb2 = bbox[:, 2] + reg[:, 2] * w
    bb3 = bbox[:, 3] + reg[:, 3] * h

    bbox[:, 0:4] = np.array([bb0, bb1, bb2, bb3]).T
    # print "bb", bbox
    return bbox


def pad(boxesA, w, h):
    boxes = boxesA.copy()  # shit, value parameter!!!

    # print 'boxes', boxes
    # print 'w,h', w, h

    tmph = boxes[:, 3] - boxes[:, 1] + 1
    tmpw = boxes[:, 2] - boxes[:, 0] + 1
    n_boxes = boxes.shape[0]

    tmph = tmph.astype(np.int32)
    tmpw = tmpw.astype(np.int32)

    # print 'tmph', tmph
    # print 'tmpw', tmpw

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

    # print "dy"  ,dy
    # print "dx"  ,dx
    # print "y "  ,y
    # print "x "  ,x
    # print "edy" ,edy
    # print "edx" ,edx
    # print "ey"  ,ey
    # print "ex"  ,ex

    # print 'boxes', boxes
    return [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph]


def convert_to_squares(bboxA):
    # convert bboxA to square
    w = bboxA[:, 2] - bboxA[:, 0]
    h = bboxA[:, 3] - bboxA[:, 1]
    l = np.maximum(w, h).T

    # print 'bboxA', bboxA
    # print 'w', w
    # print 'h', h
    # print 'l', l
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

    # print '(x,y)',x,y
    # print 'scores', scores
    # print 'reg', reg

    return bbox_out.T


def detect_face(cv_img, minsize, PNet, RNet, ONet, threshold, factor):
    img = cv_img.copy()
    img = preprocess_cvimg(img)

    factor_count = 0
    total_boxes = np.zeros((0, 9), np.float)
    points = []

    h = img.shape[0]
    w = img.shape[1]
    minl = min(h, w)

    m = 12.0 / minsize
    minl = minl * m

    # create scale pyramid
    scales = []
    while minl >= 12:
        scales.append(m * pow(factor, factor_count))
        minl *= factor
        factor_count += 1

    ###############
    # First stage
    ###############
    t1 = time.clock()

    # 1.1 run PNet
    for scale in scales:
        hs = int(np.ceil(h * scale))
        ws = int(np.ceil(w * scale))

        im_data = cv2.resize(img, (ws, hs))  # default is bilinear
        im_data = np.swapaxes(im_data, 0, 2)

        PNet.blobs['data'].reshape(1, 3, ws, hs)
        PNet.blobs['data'].data[...] = im_data
        out = PNet.forward()

        boxes = generate_bboxes(
            out['prob1'][0, 1, :, :], out['conv4-2'][0], scale, threshold[0])

        if boxes is None:
            continue

        if boxes.shape[0] > 0:
            pick = nms(boxes, 0.5, 'Union')

            if len(pick) > 0:
                boxes = boxes[pick, :]
            else:
                continue

            total_boxes = np.concatenate((total_boxes, boxes), axis=0)

    t2 = time.clock()
    #print("-->PNet cost %f seconds, processed %d pyramid scales" % ((t2-t1), len(scales)) )

    n_boxes = total_boxes.shape[0]
    # print("-->PNet outputs #total_boxes = %d" % n_boxes)

    if n_boxes < 1:
        return (None, None)

    # 1.2 run NMS
    t1 = time.clock()

    pick = nms(total_boxes, 0.7, 'Union')

    t2 = time.clock()
    #print("-->First NMS cost %f seconds, processed %d boxes" % ((t2-t1), n_boxes) )

    n_boxes = len(pick)
    #print("-->First NMS outputs %d boxes" % n_boxes )
    if n_boxes < 1:
        return (None, None)

    total_boxes = total_boxes[pick, :]

    # revise and convert to square
    regh = total_boxes[:, 3] - total_boxes[:, 1]
    regw = total_boxes[:, 2] - total_boxes[:, 0]

    t1 = total_boxes[:, 0] + total_boxes[:, 5] * regw
    t2 = total_boxes[:, 1] + total_boxes[:, 6] * regh
    t3 = total_boxes[:, 2] + total_boxes[:, 7] * regw
    t4 = total_boxes[:, 3] + total_boxes[:, 8] * regh
    t5 = total_boxes[:, 4]

    total_boxes = np.array([t1, t2, t3, t4, t5]).T

    total_boxes = convert_to_squares(total_boxes)  # convert box to square

    total_boxes[:, 0:4] = np.fix(total_boxes[:, 0:4])
    [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = pad(total_boxes, w, h)

    ###############
    # Second stage
    ###############
    t1 = time.clock()

    # 2.1 construct input for RNet
    tmp_img = np.zeros((n_boxes, 24, 24, 3))  # (24, 24, 3, n_boxes)
    for k in range(n_boxes):
        tmp = np.zeros((int(tmph[k]), int(tmpw[k]), 3))

        # print "dx[k], edx[k]:", dx[k], edx[k]
        # print "dy[k], edy[k]:", dy[k], edy[k]
        # print "img.shape", img[y[k]:ey[k]+1, x[k]:ex[k]+1].shape
        # print "tmp.shape", tmp[dy[k]:edy[k]+1, dx[k]:edx[k]+1].shape

        tmp[dy[k]:edy[k] + 1, dx[k]:edx[k] +
            1] = img[y[k]:ey[k] + 1, x[k]:ex[k] + 1]
        # print "y,ey,x,ex", y[k], ey[k], x[k], ex[k]
        # print "tmp", tmp.shape

        tmp_img[k, :, :, :] = cv2.resize(tmp, (24, 24))

    # 2.2 run RNet
    tmp_img = np.swapaxes(tmp_img, 1, 3)

    RNet.blobs['data'].reshape(n_boxes, 3, 24, 24)
    RNet.blobs['data'].data[...] = tmp_img
    out = RNet.forward()

    scores = out['prob1'][:, 1]
    pass_t = np.where(scores > threshold[1])[0]

    t2 = time.clock()
    #print("-->RNet cost %f seconds, processed %d boxes, avg time: %f seconds" % ((t2-t1), n_boxes, (t2-t1)/n_boxes) )

    n_boxes = pass_t.shape[0]
    # print("-->RNet outputs #total_boxes = %d" % n_boxes)

    if n_boxes < 1:
        return (None, None)

    scores = np.array([scores[pass_t]]).T
    total_boxes = np.concatenate((total_boxes[pass_t, 0:4], scores), axis=1)
    reg_factors = out['conv5-2'][pass_t, :].T

    # 2.3 NMS
    t1 = time.clock()
    pick = nms(total_boxes, 0.7, 'Union')
    t2 = time.clock()

    #print("-->Second NMS cost %f seconds, processed %d boxes" % ((t2-t1), n_boxes) )

    n_boxes = len(pick)
    #print("-->Second NMS outputs %d boxes" % n_boxes )

    if n_boxes < 1:
        return (None, None)

    total_boxes = total_boxes[pick, :]
    total_boxes = bbox_reg(total_boxes, reg_factors[:, pick])
    total_boxes = convert_to_squares(total_boxes)

    ###############
    # Third stage
    ###############

    # 3.1 construct input for RNet

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
    t1 = time.clock()

    tmp_img = np.swapaxes(tmp_img, 1, 3)
    ONet.blobs['data'].reshape(n_boxes, 3, 48, 48)
    ONet.blobs['data'].data[...] = tmp_img
    out = ONet.forward()

    scores = out['prob1'][:, 1]
    points = out['conv6-3']
    pass_t = np.where(scores > threshold[2])[0]

    t2 = time.clock()
    #print("-->ONet cost %f seconds, processed %d boxes, avg time: %f seconds" % ((t2-t1), n_boxes, (t2-t1)/n_boxes ))

    n_boxes = pass_t.shape[0]
    # print("-->ONet outputs #total_boxes = %d" % n_boxes)

    if n_boxes < 1:
        return (None, None)

    points = points[pass_t, :]
    scores = np.array([scores[pass_t]]).T
    total_boxes = np.concatenate(
        (total_boxes[pass_t, 0:4], scores), axis=1)
#            print "[9]:", total_boxes.shape[0]

    reg_factors = out['conv6-2'][pass_t, :].T
    w = total_boxes[:, 3] - total_boxes[:, 1] + 1
    h = total_boxes[:, 2] - total_boxes[:, 0] + 1

    points[:, 0:5] = np.tile(
        w, (5, 1)).T * points[:, 0:5] + np.tile(total_boxes[:, 0], (5, 1)).T - 1
    points[:, 5:10] = np.tile(
        h, (5, 1)).T * points[:, 5:10] + np.tile(total_boxes[:, 1], (5, 1)).T - 1

    total_boxes = bbox_reg(total_boxes, reg_factors[:, :])

    t1 = time.clock()
    pick = nms(total_boxes, 0.7, 'Min')
    t2 = time.clock()

    #print("-->Third NMS cost %f seconds, processed %d boxes" % ((t2-t1), n_boxes) )

    n_boxes = len(pick)
    #print("-->Third NMS outputs %d boxes" % n_boxes )

    if n_boxes < 1:
        return (None, None)

    total_boxes = total_boxes[pick, :]
    points = points[pick, :]

    return total_boxes, points


def initFaceDetector(caffe_model_path):
    #    minsize = 20
    #     caffe_model_path = "/home/duino/iactive/mtcnn/model"
    #    threshold = [0.6, 0.7, 0.7]
    #    factor = 0.709
    # caffe.set_mode_cpu()
    caffe.set_mode_gpu()
    PNet = caffe.Net(caffe_model_path + "/det1.prototxt",
                     caffe_model_path + "/det1.caffemodel", caffe.TEST)
    RNet = caffe.Net(caffe_model_path + "/det2.prototxt",
                     caffe_model_path + "/det2.caffemodel", caffe.TEST)
    ONet = caffe.Net(caffe_model_path + "/det3.prototxt",
                     caffe_model_path + "/det3.caffemodel", caffe.TEST)
#    return (minsize, PNet, RNet, ONet, threshold, factor)
    return (PNet, RNet, ONet)


def main():
    show_img = False
    save_dir = './fd_rlt'
    imglistfile = "./imglist.txt"
    minsize = 20
    caffe_model_path = "./model"
    threshold = [0.6, 0.7, 0.7]
    scale_factor = 0.709

    if not osp.exists(save_dir):
        os.makedirs(save_dir)

#    # caffe.set_mode_cpu()
#    caffe.set_mode_gpu()
#    PNet = caffe.Net(caffe_model_path + "/det1.prototxt",
#                     caffe_model_path + "/det1.caffemodel", caffe.TEST)
#    RNet = caffe.Net(caffe_model_path + "/det2.prototxt",
#                     caffe_model_path + "/det2.caffemodel", caffe.TEST)
#    ONet = caffe.Net(caffe_model_path + "/det3.prototxt",
#                     caffe_model_path + "/det3.caffemodel", caffe.TEST)
    PNet, RNet, ONet = initFaceDetector(caffe_model_path)

    #error = []
    fp = open(imglistfile, 'r')

    ttl_time = 0.0
    img_cnt = 0

    for line in fp:
        imgpath = line.strip()
        print("\n===>" + imgpath)
        try:
            img = cv2.imread(imgpath)
        except:
            print('failed to load image: ' + imgpath)
            continue
#        img_matlab = img.copy()
#        tmp = img_matlab[:, :, 2].copy()
#        img_matlab[:, :, 2] = img_matlab[:, :, 0]
#        img_matlab[:, :, 0] = tmp
#        img_matlab = img[:,:,(2,1,0)]
#        img_matlab = img
#        img2 = preprocess_cvimg(img)

        if img is None:
            continue

        img_cnt += 1
        t1 = time.clock()

        bboxes, points = detect_face(
            img, minsize, PNet, RNet, ONet, threshold, scale_factor)

        t2 = time.clock()
        ttl_time += t2 - t1
        print("detect_face() costs %f seconds" % (t2 - t1))

#        print('output bboxes: ' + str(bboxes))
#        print('output points: ' + str(points))
        # toc()

        if bboxes is None:
            continue

        for i in range(len(bboxes)):
            cv2.rectangle(img, (int(bboxes[i][0]), int(bboxes[i][1])), (int(
                bboxes[i][2]), int(bboxes[i][3])), (0, 255, 0), 1)

            for j in range(5):
                cv2.circle(img, (int(points[i][j]), int(
                    points[i][j + 5])), 2, (0, 0, 255), 1, -1)

        print("\n===> Processed %d images, costs %f seconds, avg time: %f seconds" % (
            img_cnt, ttl_time, ttl_time / img_cnt))

        save_name = osp.join(save_dir, osp.basename(imgpath))
        cv2.imwrite(save_name, img)

        if show_img:
            cv2.imshow('img', img)

            ch = cv2.waitKey(0) & 0xFF
            if ch == 27:
                break

    fp.close()

    if show_img:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

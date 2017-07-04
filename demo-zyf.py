#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import os.path as osp

import cv2
import numpy as np

import time

from mtcnn_detector import mtcnn_detector, draw_faces

def print_usage():
    usage = 'python %s <img-list-file> <save-dir>' % osp.basename(__file__)
    print('USAGE: ' + usage)

def main(imglistfile,
         save_dir,
         show_img = False):

    minsize = 20
    caffe_model_path = "./model"
    threshold = [0.6, 0.7, 0.7]
    scale_factor = 0.709

    if not osp.exists(save_dir):
        os.makedirs(save_dir)

    t1 = time.clock()
    detector = mtcnn_detector(caffe_model_path)
    t2 = time.clock()
    print("initFaceDetector() costs %f seconds" % (t2 - t1))

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

        if img is None:
            continue

        img_cnt += 1
        t1 = time.clock()

        bboxes, points = detector.detect_face( img, minsize,
                                              threshold, scale_factor)

        t2 = time.clock()
        ttl_time += t2 - t1
        print("detect_face() costs %f seconds" % (t2 - t1))

#        print('output bboxes: ' + str(bboxes))
#        print('output points: ' + str(points))
        # toc()

        if bboxes is None:
            continue

        draw_faces(img, bboxes, points)

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
    print_usage()

    imglistfile = "./imglist.txt"
    save_dir = './fd_rlt3'
    show_img = True

    print(sys.argv)

    if len(sys.argv)>1:
        imglistfile = sys.argv[1]

    if len(sys.argv)>2:
        save_dir = sys.argv[2]

    if len(sys.argv)>3:
        show_img = not(not(sys.argv[3]))

    main(imglistfile, save_dir, show_img)

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import os.path as osp
from skimage import io
import cv2
# import numpy as np

import time
import json
import urllib

from mtcnn_detector import MtcnnDetector, draw_faces


def print_usage():
    usage = 'python %s <img-list-file> <save-dir>' % osp.basename(__file__)
    print('USAGE: ' + usage)


def main(vggface_list_fn,
         save_dir,
         save_img=False,
         show_img=False):

    minsize = 20
    caffe_model_path = "../model"
    threshold = [0.6, 0.7, 0.7]
    scale_factor = 0.709

    if not osp.exists(save_dir):
        os.makedirs(save_dir)

    #fp_rlt = open(osp.join(save_dir, 'vggface_mtcnn_fd_rlt.json'), 'w')

    #result_list = []

    t1 = time.clock()
    detector = MtcnnDetector(caffe_model_path)
    t2 = time.clock()
    print("initFaceDetector() costs %f seconds" % (t2 - t1))

    #fp = open(lfw_list_fn, 'r')

    ttl_time = 0.0
    img_cnt = 0
    fp = open(vggface_list_fn,'r')
    all_lines = fp.readlines()
    count = 1
    #lines = all_lines[:100]
    for line in all_lines:
	data = json.loads(line)
        print count 
        count = count + 1
        #img_cnt +=1
        gt = data[u'label'][u'detect'][u'general_d'][u'bbox'][0][u'pts']
        pose = data[u'label'][u'detect'][u'general_d'][u'bbox'][0][u'pose']
	isExists=os.path.exists('./dataset/vgg_face_dataset/vgg_face/%s' % data[u'url'].split('/')[-2])
	if not isExists: 
            os.makedirs('./dataset/vgg_face_dataset/vgg_face/%s' % data[u'url'].split('/')[-2])
#	urllib.urlretrieve(data[u'url'],'./dataset/vgg_face_dataset/vgg_face/%s/%s' % (data[u'url'].split('/')[-2],data[u'url'].split('/')[-1]))
	#print data[u'url'].split('/')[-1][:-4]
        result_list = []	
	resultpath = '%s/%s' % (data[u'url'].split('/')[-2],data[u'url'].split('/')[-1][:-4])
        result_name = resultpath + '.json'
        imgpath = './dataset/vgg_face_dataset/vgg_face/%s/%s' % (data[u'url'].split('/')[-2],data[u'url'].split('/')[-1])
        #print("\n===>" + imgpath)
      
        isExists2=os.path.exists('./result_json/%s' % data[u'url'].split('/')[-2])
        isExists3=os.path.exists(osp.join(save_dir,result_name))
        if not isExists2:
            os.makedirs('./result_json/%s' % data[u'url'].split('/')[-2])
        if isExists3:
            continue
        
        print("\n===>" + imgpath)
        fp_rlt = open(osp.join(save_dir,result_name),'w')
        #id = 'unkown' if len(splits) < 2 else splits[1]
	id = data[u'facecluster']
        rlt = {}
        rlt["filename"] = imgpath
        rlt["faces"] = []
        rlt['face_count'] = 0
        rlt['id'] = id
        rlt['gt'] = gt
        rlt['pose'] = pose

        try:	
	    img = io.imread(data[u'url'])
        #print "img.shape",img.shape
        except:
            print('failed to load image: ' + imgpath)
            rlt["message"] = "failed to load"
            result_list.append(rlt)
            json.dump(result_list,fp_rlt,indent=4)
            fp_rlt.close()
            continue
        #if img is None:
	if len(img.shape) != 3:
	    print('failed to load image: ' + imgpath)
            rlt["message"] = "failed to load"
            result_list.append(rlt)
            json.dump(result_list,fp_rlt,indent=4)
            fp_rlt.close()
            continue

        img_cnt += 1
        t1 = time.clock()

        bboxes, points = detector.detect_face(img, minsize,
                                              threshold, scale_factor)

        t2 = time.clock()
        ttl_time += t2 - t1
        print("detect_face() costs %f seconds" % (t2 - t1))

        if len(bboxes) > 0:
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
        result_list.append(rlt)

#        print('output bboxes: ' + str(bboxes))
#        print('output points: ' + str(points))
        # toc()

        if bboxes is None:
            json.dump(result_list,fp_rlt,indent=4)
            fp_rlt.close()
            continue

        print("\n===> Processed %d images, costs %f seconds, avg time: %f seconds" % (
            img_cnt, ttl_time, ttl_time / img_cnt))

        if save_img or show_img:
            draw_faces(img, bboxes, points)

        if save_img:
            save_name = osp.join(save_dir, osp.basename(imgpath))
            cv2.imwrite(save_name, img)

        if show_img:
            cv2.imshow('img', img)

            ch = cv2.waitKey(0) & 0xFF
            if ch == 27:
                break

        json.dump(result_list, fp_rlt, indent=4)
        fp_rlt.close()
    fp.close()

    if show_img:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    print_usage()

#    lfw_list_fn = "./lfw_list_part.txt"
    vggface_list_fn = "./summary_atflow_format.txt"
    #vggface_list_fn = "./vggface-test.txt"
#    lfw_list_fn = "lfw_list_mtcnn.txt"
    save_dir = './result_json'
#    lfw_root = '/disk2/data/FACE/LFW/LFW'
    #lfw_root = r'C:\zyf\dataset\lfw'

    print(sys.argv)

    if not osp.exists(save_dir):
        os.makedirs(save_dir)

    if len(sys.argv) > 1:
        lfw_list_fn = sys.argv[1]

    if len(sys.argv) > 2:
        save_dir = sys.argv[2]

    if len(sys.argv) > 3:
        show_img = not(not(sys.argv[3]))

#    main(lfw_list_fn, lfw_root, save_dir, save_img=True, show_img=True)
    main(vggface_list_fn, save_dir, save_img=False, show_img=False)

import numpy as np 
import tensorflow as tf 
import argparse
import sys
import os
import time
import cv2 as cv 
from PIL import Image
from matplotlib import pyplot as plt

from label_utils import label_map_util

FLAGS = None

class ObjectDetector(object):

    def __init__(self,model_path,label_path):
        self.model_path=model_path
        self.graph = self.create_graph(self.model_path)
        self.sess = tf.Session(graph=self.graph)
        self.category_index = label_map_util.create_category_index_from_labelmap(label_path, use_display_name=True)


    @staticmethod
    def create_graph(model_path):
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(model_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        return detection_graph

    #前馈传播，计算检验的目标分类，框，分数信息
    def run_inference_for_single_image(self, image):
        # 从图中取出所有算子的名字
        ops = self.graph.get_operations()
        all_tensor_names = {output.name for op in ops for output in op.outputs}
        # 提取需要计算的算子
        tensor_dict = {}
        for key in ['num_detections', 'detection_boxes', 'detection_scores','detection_classes', 'detection_masks']:
            tensor_name = key + ':0'
            if tensor_name in all_tensor_names:
                    tensor_dict[key] = self.graph.get_tensor_by_name(tensor_name)
        # 输入算子
        image_tensor = self.graph.get_tensor_by_name('image_tensor:0')
        # 前馈传播计算需要计算的算子
        start_time = time.time()
        output_dict = self.sess.run(tensor_dict, feed_dict={image_tensor: np.expand_dims(image, 0)})
        end_time = time.time()
        print("run time:", end_time - start_time)
        return self.deal_box_score_info(output_dict,image)

    
    def deal_box_score_info(self,output_dict,image):
        result = []
        (image_height,image_width) = image.shape[0],image.shape[1]
        # 检测到的框数
        num = int(output_dict['num_detections'][0])
        output_dict['num_detections'] = num
        output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)[:num]
        output_dict['detection_boxes'] = output_dict['detection_boxes'][0][:num]
        output_dict['detection_scores'] = output_dict['detection_scores'][0][:num]

        for i in range(num):
            box_info = {}
            box_info['cls_id'] = output_dict['detection_classes'][i]
            box_info['cls_name'] = self.category_index[box_info['cls_id']]['name']
            box_info['score'] = output_dict['detection_scores'][i]
            box_info['box_ymin'] = int(output_dict['detection_boxes'][i][0]*image_height)
            box_info['box_xmin'] = int(output_dict['detection_boxes'][i][1]*image_width)
            box_info['box_ymax'] = int(output_dict['detection_boxes'][i][2]*image_height)
            box_info['box_xmax'] = int(output_dict['detection_boxes'][i][3]*image_width)

            result.append(box_info)

        return result


def isimage(path):
  return os.path.splitext(path)[1].lower() in ['.jpg', '.png', '.jpeg']


def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

def main():
    #构建对象识别器
    obj_detector = ObjectDetector(FLAGS.model_frozen,FLAGS.label_path)

    # 图片信息
    image_paths = []
    if os.path.isfile(FLAGS.image_path):
        image_paths.append(FLAGS.image_path)
    else:
        for file_or_dir in os.listdir(FLAGS.image_path):
            file_path = os.path.join(FLAGS.image_path, file_or_dir)
            if os.path.isfile(file_path) and isimage(file_path):
                image_paths.append(file_path)
    print(image_paths)


    for image_path in image_paths:
        # prepare data
        image = Image.open(image_path)
        image_np = load_image_into_numpy_array(image)
        print(image_np.shape)
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        #image_np_expanded = np.expand_dims(image_np, axis=0)

        # Actual detection.
        box_infos = obj_detector.run_inference_for_single_image(image_np)

        for i in range(len(box_infos)):
            box_info = box_infos[i]
            cls_name = box_info['cls_name']
            score = box_info['score']
            box_ymin = box_info['box_ymin']
            box_xmin = box_info['box_xmin']
            box_ymax = box_info['box_ymax']
            box_xmax = box_info['box_xmax']
            cv.rectangle(image_np, (box_xmin,box_ymin), (box_xmax,box_ymax), (0,255,0),3)
            text = "%s:%.2f" % (cls_name,score)
            cv.putText(image_np, text, (box_xmin,box_ymin-4),cv.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (255,0,0))

        plt.figure(figsize=(12, 8)) # Size, in inches
        plt.imshow(image_np)
        plt.show()

    obj_detector.sess.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str,
                      default='test_images/',
                      help='image path')
    parser.add_argument('--model_frozen', type=str,
                      default='../model/ssd_mobilenet_v1_coco_2018_01_28/frozen_inference_graph.pb',
                      help='model path')
    parser.add_argument('--label_path', type=str,
                      default='label_utils/mscoco_label_map.pbtxt',
                      help='label path')
    FLAGS, unparsed = parser.parse_known_args()
    main()

            
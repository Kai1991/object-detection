import cv2 as cv
import numpy as np 
import tensorflow as tf 
from object_detector import ObjectDetector
import argparse

FLAGS = None


def visualize_info(frame, box_infos):
    lines_visualize = np.zeros_like(frame)

    for i in range(len(box_infos)):
        box_info = box_infos[i]
        cls_name = box_info['cls_name']
        score = box_info['score']
        box_ymin = box_info['box_ymin']
        box_xmin = box_info['box_xmin']
        box_ymax = box_info['box_ymax']
        box_xmax = box_info['box_xmax']
        cv.rectangle(lines_visualize, (box_xmin,box_ymin), (box_xmax,box_ymax), (0,255,0),3)
        text = "%s:%.2f" % (cls_name,score)
        cv.putText(lines_visualize, text, (box_xmin,box_ymin-4),cv.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (255,0,0))
    return lines_visualize

def main():
    #物体识别器
    obj_detector = ObjectDetector(FLAGS.model_frozen,FLAGS.label_path)
    # 1.读取视频
    cap = cv.VideoCapture('./test_video/input.mp4')
    while(cap.isOpened()):
        # 处理每一帧数据
        _, frame = cap.read()
        
        box_infos = obj_detector.run_inference_for_single_image(frame)

        #绘制车道线
        box_visualize = visualize_info(frame, box_infos)
        #显示到原来图层中
        output = cv.addWeighted(frame, 0.9, box_visualize, 1, 1)
        cv.imshow("output", output)
        if cv.waitKey(10) & 0xFF == ord('q'):
            break

    #释放资源
    cap.release()
    cv.destroyAllWindows()


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

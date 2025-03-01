import tensorflow as tf
import numpy as np
import cv2

# 加载预训练的 SSD 模型（SSD MobileNet V2）
model = tf.saved_model.load('saved_model')

def load_model(model_path):
    return tf.saved_model.load(model_path)


def run_inference_for_single_image(model, image):
    # 增加一个批次维度并进行推理
    image = np.asarray(image)
    input_tensor = tf.convert_to_tensor(image)
    input_tensor = input_tensor[tf.newaxis, ...]
    model_fn = model.signatures['serving_default']
    output_dict = model_fn(input_tensor)

    # 准备返回的检测结果
    output_dict = {key: value.numpy() for key, value in output_dict.items()}
    return output_dict


def draw_boxes(image, boxes, class_ids, scores, threshold=0.5):
    for i in range(len(boxes)):
        if scores[i] > threshold:
            # 获取每个框的坐标
            ymin, xmin, ymax, xmax = boxes[i]
            (left, right, top, bottom) = (xmin * image.shape[1], xmax * image.shape[1],
                                          ymin * image.shape[0], ymax * image.shape[0])
            # 绘制矩形框
            cv2.rectangle(image, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)
            # 绘制标签和得分
            label = str(int(class_ids[i]))
            score = str(round(scores[i], 2))
            cv2.putText(image, f"{label} {score}", (int(left), int(top) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)


# 打开视频文件
cap = cv2.VideoCapture("testvideo.mp4")

# 获取视频的基本信息
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))



fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 mp4v编解码器
out = cv2.VideoWriter("output_video.mp4", fourcc, fps, (frame_width, frame_height))


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 执行推理
    output_dict = run_inference_for_single_image(model, frame)

    # 获取边框、类别和分数
    boxes = output_dict['detection_boxes'][0]
    class_ids = output_dict['detection_classes'][0]
    scores = output_dict['detection_scores'][0]

    # 绘制检测框
    draw_boxes(frame, boxes, class_ids, scores)

    # 写入处理过的视频帧
    out.write(frame)

# 释放资源
cap.release()
out.release()
cv2.destroyAllWindows()

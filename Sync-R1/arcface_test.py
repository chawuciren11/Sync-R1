from PIL import Image
from mtcnn_pytorch.src import detect_faces  # 导入MTCNN检测工具
from mtcnn_pytorch.src.matlab_cp2torm import get_similarity_transform_for_cv2  # 对齐工具
import cv2
import numpy as np

def align_face(img_path, target_size=(112, 112)):
    """使用MTCNN检测人脸并对齐"""
    # 读取图像
    img = Image.open(img_path).convert('RGB')
    # 检测人脸边界框和关键点（5个关键点：左眼、右眼、鼻子、左嘴角、右嘴角）
    bounding_boxes, landmarks = detect_faces(img)
    
    if len(bounding_boxes) == 0:
        raise ValueError(f"未在 {img_path} 中检测到人脸")
    
    # 取第一个检测到的人脸（最高置信度）
    landmarks = landmarks[0].reshape(2, 5).T  # 转换为 (5, 2) 格式（x,y坐标）
    
    # 目标对齐关键点（标准112x112人脸的关键点坐标）
    src = np.array([
        [30.2946, 51.6963],   # 左眼
        [65.5318, 51.5014],   # 右眼
        [48.0252, 71.7366],   # 鼻子
        [33.5493, 92.3655],   # 左嘴角
        [62.7299, 92.2041]    # 右嘴角
    ], dtype=np.float32)
    
    # 计算相似变换矩阵（对齐人脸）
    transform = get_similarity_transform_for_cv2(landmarks, src)
    # 应用变换，得到对齐后的人脸
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    aligned_face = cv2.warpAffine(img_cv, transform, target_size)
    # 转换回PIL格式（RGB通道）
    aligned_face = Image.fromarray(cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB))
    return aligned_face

# 示例：对齐两张人脸图像
aligned_face1 = align_face("/share/project/emllm_mnt.1d/mnt/hpfs/baaiei/daigaole/code/UnicR1/showo/tmp_result/images/adrien_brody/part_1.png")  # 第一张人脸路径
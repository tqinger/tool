import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from xml.dom.minidom import parseString
from PIL import Image

def generate_xml(filename, bbox_list, img_size, class_names, dest_folder):
    """
    生成Pascal VOC格式的XML标注文件。
    """
    folder_name = os.path.basename(dest_folder)
    img_name = filename + '.jpg'
    xml_file = filename + '.xml'

    root = ET.Element("annotation")
    ET.SubElement(root, "folder").text = folder_name
    ET.SubElement(root, "filename").text = img_name

    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(img_size[1])
    ET.SubElement(size, "height").text = str(img_size[0])
    ET.SubElement(size, "depth").text = str(img_size[2])

    for bbox, cls_name in bbox_list:
        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = cls_name
        bndbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bndbox, "xmin").text = str(bbox[0])
        ET.SubElement(bndbox, "ymin").text = str(bbox[1])
        ET.SubElement(bndbox, "xmax").text = str(bbox[2])
        ET.SubElement(bndbox, "ymax").text = str(bbox[3])

    tree = ET.ElementTree(root)
    xml_str = parseString(ET.tostring(root)).toprettyxml(indent="   ")
    with open(os.path.join(dest_folder, xml_file), 'w') as f:
        f.write(xml_str)


def extract_bboxes_from_segmentation(seg_img):
    """
    从分割图像中提取边界框。
    """
    bboxes = []
    target = Image.open(seg_img)
    gt2D = np.array(target)
    y_indices, x_indices = np.where(gt2D > 0)
    xmin, xmax = np.min(x_indices), np.max(x_indices)
    ymin, ymax = np.min(y_indices), np.max(y_indices)

    bboxes.append(((xmin, ymin, xmax, ymax), "shetou"))
    return bboxes

if __name__ == "__main__":
    dest_folder = r'C:\Users\Public\data\she\514_yuantu\VOCdevkit\VOC2012\Annotations'  # 保存XML文件的目录
    img_list = os.listdir(r"C:\Users\Public\data\she\514_yuantu\VOCdevkit\VOC2012\SegmentationClass")
    for img_name in img_list:
        seg_img_path = os.path.join(r"C:\Users\Public\data\she\514_yuantu\VOCdevkit\VOC2012\SegmentationClass",img_name)  # 分割标注图像路径

        class_names = ['background', 'shetou']  # 类别名列表，按照实际情况填写

        # 读取分割图像
        seg_img = cv2.imread(seg_img_path, cv2.IMREAD_GRAYSCALE)
        img_size = seg_img.shape + (1,)  # 假设是单通道图像

        # 提取边界框信息
        bboxes = extract_bboxes_from_segmentation(seg_img_path)  # 从1开始去除背景

        # 生成XML
        filename = os.path.splitext(os.path.basename(seg_img_path))[0]
        generate_xml(filename, bboxes, img_size, class_names, dest_folder)

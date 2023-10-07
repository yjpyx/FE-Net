import os
import json
import xml.etree.ElementTree as ET


def convert_coco_to_voc(coco_path, img_dir, save_dir):
    with open(coco_path, 'r') as f:
        coco_data = json.load(f)

    categories = coco_data['categories']
    images = coco_data['images']
    annotations = coco_data['annotations']

    for image in images:
        image_id = image['id']
        image_file_name = image['file_name']
        image_width = image['width']
        image_height = image['height']

        # 创建VOC格式的XML文件
        root = ET.Element('annotation')

        folder = ET.SubElement(root, 'folder')
        folder.text = 'HRSID'

        filename = ET.SubElement(root, 'filename')
        filename.text = image_file_name

        size = ET.SubElement(root, 'size')
        width = ET.SubElement(size, 'width')
        width.text = str(image_width)
        height = ET.SubElement(size, 'height')
        height.text = str(image_height)
        depth = ET.SubElement(size, 'depth')
        depth.text = '3'  # 假设图片是RGB三通道

        objects = []
        for annotation in annotations:
            if annotation['image_id'] == image_id:
                object_info = {
                    'category_id': annotation['category_id'],
                    'bbox': annotation['bbox']
                }
                objects.append(object_info)

        for obj in objects:
            category_id = obj['category_id']
            bbox = obj['bbox']

            category = next((cat for cat in categories if cat['id'] == category_id), None)
            if category:
                object_name = category['name']

                xmin, ymin, width, height = bbox
                xmax = xmin + width
                ymax = ymin + height

                object_elem = ET.SubElement(root, 'object')
                name = ET.SubElement(object_elem, 'name')
                name.text = object_name

                bbox = ET.SubElement(object_elem, 'bndbox')
                xmin_elem = ET.SubElement(bbox, 'xmin')
                xmin_elem.text = str(int(xmin))
                ymin_elem = ET.SubElement(bbox, 'ymin')
                ymin_elem.text = str(int(ymin))
                xmax_elem = ET.SubElement(bbox, 'xmax')
                xmax_elem.text = str(int(xmax))
                ymax_elem = ET.SubElement(bbox, 'ymax')
                ymax_elem.text = str(int(ymax))

        xml_file = os.path.join(save_dir, os.path.splitext(image_file_name)[0] + '.xml')
        tree = ET.ElementTree(root)
        tree.write(xml_file, encoding='utf-8')

    print('转换完成！')


if __name__ == "__main__":
    coco_path = 'HRSID_JPG/annotations/train_test2017.json'  # COCO数据路径
    img_dir = 'HRSID_JPG/JPEGImages/'  # 图片文件夹路径
    save_dir = 'HRSID_VOC'  # 保存VOC格式数据的文件夹路径

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    convert_coco_to_voc(coco_path, img_dir, save_dir)

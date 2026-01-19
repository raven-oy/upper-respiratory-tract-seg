import os
import numpy as np
from PIL import Image
import shutil

def process_images():
    train_dir = 'dataset/train'
    gt_dir = 'dataset/train_GT'

    # 获取所有TIF文件
    tif_files = [f for f in os.listdir(train_dir) if f.endswith('.TIF') and not f.startswith('ISIC_')]
    tif_files.sort()  # 排序

    print(f"找到 {len(tif_files)} 个新TIF文件需要处理")

    for i, tif_file in enumerate(tif_files):
        # 生成新的ISIC文件名
        new_index = 6 + i  # 从0000006开始
        new_name = f"ISIC_{new_index:07d}"

        old_path = os.path.join(train_dir, tif_file)
        new_image_path = os.path.join(train_dir, f"{new_name}.TIF")
        new_gt_path = os.path.join(gt_dir, f"{new_name}_segmentation.png")

        # 重命名图像文件
        shutil.move(old_path, new_image_path)
        print(f"重命名: {tif_file} -> {new_name}.TIF")

        # 处理图像生成标注
        try:
            # 读取图像
            img = Image.open(new_image_path)

            # 转换为numpy数组
            img_array = np.array(img)

            # 如果是多通道，取第一个通道或转换为灰度
            if len(img_array.shape) > 2:
                img_array = img_array[:, :, 0]  # 取第一个通道

            # 确保是8bit
            if img_array.dtype != np.uint8:
                img_array = (img_array / img_array.max() * 255).astype(np.uint8)

            # 阈值处理：>86的像素设为255，否则0
            threshold = 86
            binary_mask = (img_array > threshold).astype(np.uint8) * 255

            # 保存为PNG
            mask_img = Image.fromarray(binary_mask, mode='L')
            mask_img.save(new_gt_path)
            print(f"生成标注: {new_name}_segmentation.png")

        except Exception as e:
            print(f"处理 {tif_file} 时出错: {e}")

if __name__ == "__main__":
    process_images()
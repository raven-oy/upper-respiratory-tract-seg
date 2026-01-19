import os
import shutil
import random

def split_dataset():
    train_dir = 'dataset/train'
    train_gt_dir = 'dataset/train_GT'
    valid_dir = 'dataset/valid'
    valid_gt_dir = 'dataset/valid_GT'
    test_dir = 'dataset/test'
    test_gt_dir = 'dataset/test_GT'

    # 获取所有训练文件
    image_files = [f for f in os.listdir(train_dir) if f.endswith('.TIF')]
    image_files.sort()

    total = len(image_files)
    print(f"总共有 {total} 个训练样本")

    # 分配比例：70% 训练，20% 验证，10% 测试
    train_count = int(total * 0.7)
    valid_count = int(total * 0.2)
    test_count = total - train_count - valid_count

    print(f"分配: 训练 {train_count} 个, 验证 {valid_count} 个, 测试 {test_count} 个")

    # 随机打乱
    random.seed(42)  # 固定种子保证可重复
    random.shuffle(image_files)

    # 分配文件
    train_files = image_files[:train_count]
    valid_files = image_files[train_count:train_count + valid_count]
    test_files = image_files[train_count + valid_count:]

    def move_files(file_list, src_img_dir, src_gt_dir, dst_img_dir, dst_gt_dir):
        for img_file in file_list:
            # 移动图像
            shutil.move(os.path.join(src_img_dir, img_file), os.path.join(dst_img_dir, img_file))

            # 移动对应的标注
            gt_file = img_file.replace('.TIF', '_segmentation.png')
            shutil.move(os.path.join(src_gt_dir, gt_file), os.path.join(dst_gt_dir, gt_file))

    # 移动到验证集
    move_files(valid_files, train_dir, train_gt_dir, valid_dir, valid_gt_dir)
    print(f"已移动 {len(valid_files)} 个样本到验证集")

    # 移动到测试集
    move_files(test_files, train_dir, train_gt_dir, test_dir, test_gt_dir)
    print(f"已移动 {len(test_files)} 个样本到测试集")

    # 统计最终数量
    final_train = len([f for f in os.listdir(train_dir) if f.endswith('.TIF')])
    final_valid = len([f for f in os.listdir(valid_dir) if f.endswith('.TIF')])
    final_test = len([f for f in os.listdir(test_dir) if f.endswith('.TIF')])

    print("
最终数据分布:"    print(f"训练集: {final_train} 个样本")
    print(f"验证集: {final_valid} 个样本")
    print(f"测试集: {final_test} 个样本")

if __name__ == "__main__":
    split_dataset()
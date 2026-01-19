import os
import shutil
import random

def copy_validation_test():
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
    print(f"训练集中有 {total} 个样本")

    # 选择一些样本复制到验证和测试集
    # 验证集：复制前10个
    # 测试集：复制接下来5个

    valid_count = min(10, total // 3)
    test_count = min(5, (total - valid_count) // 2)

    valid_files = image_files[:valid_count]
    test_files = image_files[valid_count:valid_count + test_count]

    print(f"将复制 {len(valid_files)} 个样本到验证集")
    print(f"将复制 {len(test_files)} 个样本到测试集")

    def copy_files(file_list, src_img_dir, src_gt_dir, dst_img_dir, dst_gt_dir):
        for img_file in file_list:
            # 复制图像
            shutil.copy2(os.path.join(src_img_dir, img_file), os.path.join(dst_img_dir, img_file))

            # 复制对应的标注
            gt_file = img_file.replace('.TIF', '_segmentation.png')
            shutil.copy2(os.path.join(src_gt_dir, gt_file), os.path.join(dst_gt_dir, gt_file))

    # 复制到验证集
    copy_files(valid_files, train_dir, train_gt_dir, valid_dir, valid_gt_dir)
    print(f"✅ 已复制 {len(valid_files)} 个样本到验证集")

    # 复制到测试集
    copy_files(test_files, train_dir, train_gt_dir, test_dir, test_gt_dir)
    print(f"✅ 已复制 {len(test_files)} 个样本到测试集")

    # 统计最终数量
    final_train = len([f for f in os.listdir(train_dir) if f.endswith('.TIF')])
    final_valid = len([f for f in os.listdir(valid_dir) if f.endswith('.TIF')])
    final_test = len([f for f in os.listdir(test_dir) if f.endswith('.TIF')])

    print("
最终数据分布:"    print(f"训练集: {final_train} 个样本")
    print(f"验证集: {final_valid} 个样本")
    print(f"测试集: {final_test} 个样本")

if __name__ == "__main__":
    copy_validation_test()
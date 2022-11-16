# -*- coding: utf-8 -*-
"""
# @file name  : 1_split_dataset.py
# @author     : zyy
# @date       : 2019-09-07 10:08:00
# @brief      : 将数据集划分为训练集，验证集，测试集
"""

import os
import random
import sys
import shutil

# 如果我们创建的路径不存在，则创建该路径
def makedir(new_dir):
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)


if __name__ == '__main__':

    random.seed(1)

    # 路径拼接，创建一系列文件夹
    dataset_dir = os.path.join("..", "..", "data", "RMB_data")
    # print(dataset_dir)
    split_dir = os.path.join("..", "..", "data", "rmb_split")
    # print(split_dir)
    train_dir = os.path.join(split_dir, "train")
    # print(train_dir)
    valid_dir = os.path.join(split_dir, "valid")
    # print(valid_dir)
    test_dir = os.path.join(split_dir, "test")
    # print(test_dir)

    # 训练集、验证集、测试集的划分
    train_pct = 0.8
    valid_pct = 0.1
    test_pct = 0.1

    # root指的是当前所在的文件夹路径，dirs是当前文件夹路径下的文件夹列表，files是当前文件夹路径下的文件列表
    for root, dirs, files in os.walk(dataset_dir):
        # 遍历每一个文件夹列表,获取文件夹下的每一个文件
        for sub_dir in dirs:

            imgs = os.listdir(os.path.join(root, sub_dir))
            # 将.jpg文件过滤出来
            imgs = list(filter(lambda x: x.endswith('.jpg'), imgs))
            # 打乱顺序
            random.shuffle(imgs)
            # 图片数量
            img_count = len(imgs)
            # 训练集数量
            train_point = int(img_count * train_pct)
            # 验证集数量
            valid_point = int(img_count * (train_pct + valid_pct))

            if img_count == 0:
                print("{}目录下，无图片，请检查".format(os.path.join(root, sub_dir)))
                sys.exit(0)
            # 将图片存入对应的文件夹中
            for i in range(img_count):
                if i < train_point:
                    out_dir = os.path.join(train_dir, sub_dir)
                elif i < valid_point:
                    out_dir = os.path.join(valid_dir, sub_dir)
                else:
                    out_dir = os.path.join(test_dir, sub_dir)

                # 创建对应的路径
                makedir(out_dir)

                target_path = os.path.join(out_dir, imgs[i])
                src_path = os.path.join(dataset_dir, sub_dir, imgs[i])

                # 将前者复制到后者中
                shutil.copy(src_path, target_path)

            print('Class:{}, train:{}, valid:{}, test:{}'.format(sub_dir, train_point, valid_point-train_point,
                                                                 img_count-valid_point))
            print("已在 {} 创建划分好的数据\n".format(out_dir))



# -*- coding: utf-8 -*-
# @File name: lesson-2.py
# @Time : 2022-9-19 11:01
# @Author : zyy
# @Brief : 张量操作与线性回归

import numpy as np
import torch

# ---------------------------------------torch.cat拼接----------------------------------------------

# flag = True
flag = False

if flag:
    t = torch.ones((2, 3))
    print(t)

    t_0 = torch.cat([t, t], dim=0)
    t_1 = torch.cat([t, t], dim=1)

    print("t_0:{} shape:{}\nt_1:{} shape:{}".format(t_0, t_0.shape, t_1, t_1.shape))

# ---------------------------------------torch.stack拼接----------------------------------------------
# flag = True
flag = False

if flag:
    t = torch.ones((2, 3))
    print(t)

    t_stack0 = torch.stack([t, t], dim=0)
    t_stack1 = torch.stack([t, t], dim=1)

    print("\nt_stack0:{} shape:{}\nt_stack1:{} shape:{}".format(t_stack0, t_stack0.shape, t_stack1, t_stack1.shape))

# ---------------------------------------torch.chunk切分----------------------------------------------
flag = True
# flag = False

if flag:
    a = torch.ones((2, 5))
    list_of_tensors = torch.chunk(a, dim=1, chunks=2)

    for idx, t in enumerate(list_of_tensors):
        print("第{}个张量：{}，shape is {}".format(idx+1, t, t.shape))

# ---------------------------------------torch.split切分----------------------------------------------
# flag = True
flag = False

if flag:
    t = torch.ones((2, 5))

    list_of_tensors = torch.split(t, 2, dim=1)
    for idx, t in enumerate(list_of_tensors):
        print("第{}个张量：{}， shape is {}".format(idx+1, t, t.shape))

# ---------------------------------------torch.index_select索引----------------------------------------------
# flag = True
flag = False

if flag:
    t = torch.randint(0, 9, size=(3, 3))
    print(t, t.shape)

    idx = torch.tensor([0, 2],dtype=torch.long)   # float会报错
    t_select = torch.index_select(t, dim=1, index=idx)
    print("t:\n{}\nt_select:\n{}".format(t, t_select))

# ---------------------------------------torch.masked_select索引----------------------------------------------
# flag = True
flag = False

if flag:
    t = torch.randint(0, 9, size=(3, 3))
    mask = t.ge(5)  # ge is mean greater than or equal/  gt: greater than  le  lt
    t_select = torch.masked_select(t, mask)
    print("t:\n{}\nmask:\n{}\nt_select:\n{}".format(t, mask, t_select))

# ---------------------------------------torch.reshape变换----------------------------------------------
# flag = True
flag = False

if flag:
    t = torch.randperm(8)
    t_reshape = torch.reshape(t, (2, 4))     # -1
    print("t:\n{}\nt_reshape:\n{}".format(t, t_reshape))

    t[0] = 1024
    print("t:\n{}\nt_reshape:\n{}".format(t, t_reshape))
    print("t.data 内存地址：{}".format(id(t.data)))
    print("t_reshape.data 内存地址：{}".format(id(t_reshape.data)))

# ---------------------------------------torch.transpose变换----------------------------------------------
# flag = True
flag = False

if flag:
    t = torch.rand((2, 3, 4))
    print(t, t.shape)

    t_transpose = torch.transpose(t, dim0=1, dim1=2)  # 图像预处理中会遇到 c*h*w
    print("t shape:{}\nt_transpose shape:{}".format(t.shape, t_transpose.shape))

# ---------------------------------------torch.squeeze变换----------------------------------------------
# flag = True
flag = False

if flag:
    t = torch.rand((1, 2, 3, 1))
    print(t)
    t_sq = torch.squeeze(t)
    t_0 = torch.squeeze(t, dim=0)
    t_1 = torch.squeeze(t, dim=1)
    print(t.shape)
    print(t_sq.shape)
    print(t_0.shape)
    print(t_1.shape)

# ---------------------------------------torch.add数学操作----------------------------------------------
# flag = True
flag = False

if flag:
    t_0 = torch.randn((3, 3))
    t_1 = torch.ones_like(t_0)
    t_add = torch.add(t_0, 10, t_1)

    print("t_0:\n{}\nt_1:\n{}\nt_add:\n{}".format(t_0, t_1, t_add))

# ---------------------------------------线性回归----------------------------------------------
# import torch
import matplotlib.pyplot as plt
torch.manual_seed(10)

lr = 0.1  # 学习率

# 创建训练数据
x = torch.rand(20, 1) * 10  # x data (tensor), shape=(20, 1)
y = 2*x + (5 + torch.randn(20, 1))  # y data (tensor), shape=(20, 1)

# 构建线性回归参数
w = torch.randn((1), requires_grad=True)    # 初始化
b = torch.zeros((1), requires_grad=True)

for iteration in range(1000):

    # 前向传播
    wx = torch.mul(w, x)
    y_pred = torch.add(wx, b)

    # 计算 MSE Loss
    loss = (0.5 * (y - y_pred) ** 2).mean()   # 均方差

    # 反向传播
    loss.backward()

    # 更新参数
    b.data.sub_(lr * b.grad)
    w.data.sub_(lr * w.grad)

    # 绘图
    if iteration % 20 == 0:

        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), y_pred.data.numpy(), 'r', lw=5)
        plt.text(2, 20, 'loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color': 'red'})
        plt.xlim(1.5, 10)
        plt.ylim(0, 28)
        plt.title("Iteration: {} b: {}".format(iteration, w.data.numpy(), b.data.numpy()))
        plt.pause(0.5)

        if loss.data.numpy() < 1:
            break
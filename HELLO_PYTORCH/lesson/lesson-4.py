# -*- coding: utf-8 -*-
# @File name: lesson-4.py
# @Time : 2022-10-8 15:55
# @Author : zyy
# @Brief : autograd

import torch
torch.manual_seed(10)    # 设置CPU生成随机数的种子，方便下次复现实验结果。

# ================================================== retain_graph ======================================================
# flag = True
flag = False
if flag:
    w = torch.tensor([1.], requires_grad=True)   # w = 1
    x = torch.tensor([2.], requires_grad=True)   # x = 2

    a = torch.add(w, x)
    b = torch.add(w, 1)
    y = torch.mul(a, b)

    # y.backward(retain_graph=True)       # 反向传播
    y.backward()
    print(w.grad)

# ================================================== grad_tensors ======================================================
# flag = True
flag = False
if flag:
    w = torch.tensor([1.], requires_grad=True)   # w = 1
    x = torch.tensor([2.], requires_grad=True)   # x = 2

    a = torch.add(w, x)   # retain_grad()
    b = torch.add(w, 1)

    y0 = torch.mul(a, b)  # y0 = (x+w) * (w+1)
    y1 = torch.add(a, b)  # y1 = (x+w) + (w+1)   dy1/dw = 2

    loss = torch.cat([y0, y1], dim=0)   # [y0, y1]
    grad_tensors = torch.tensor([1., 2.])  # 多个梯度之间权重的设置

    loss.backward(gradient=grad_tensors)  # gradient 传入 torch.autograd.backward()中的grad_tensors
    print(w.grad)

# ================================================== grad_tensors ======================================================
# flag = True
flag = False
if flag:

    x = torch.tensor([3.], requires_grad=True)
    y = torch.pow(x, 2)   # y = x**2

    grad_1 = torch.autograd.grad(y, x, create_graph=True)  # grad_1 = dy/dx = 2x = 2*3 = 6
    print(grad_1)

    grad_2 = torch.autograd.grad(grad_1[0], x)  # grad_2 = d(dy/dx)/dx = d(2x)/2x = 2
    print(grad_2)

# ================================================== tips: 1 ======================================================
# flag = True
flag = False
if flag:
    w = torch.tensor([1.], requires_grad=True)   # w = 1
    x = torch.tensor([2.], requires_grad=True)   # x = 2

    for i in range(4):     # 4次循环，梯度叠加
        a = torch.add(w, x)
        b = torch.add(w, 1)
        y = torch.mul(a, b)

        y.backward()
        print(w.grad)

        w.grad.zero_()   # 对梯度手动清零，下划线表示原位（原地）操作

# ================================================== tips: 2 ======================================================
# flag = True
flag = False
if flag:
    w = torch.tensor([1.], requires_grad=True)   # w = 1
    x = torch.tensor([2.], requires_grad=True)   # x = 2

    a = torch.add(w, x)
    b = torch.add(w, 1)
    y = torch.mul(a, b)

    print(a.requires_grad, b.requires_grad, y.requires_grad)

# ================================================== tips: 3 ======================================================
flag = True
# flag = False
if flag:

    a = torch.ones((1, ))
    print(id(a), a)

    a = a + torch.ones((1, ))   # 开辟新的内存地址
    print(id(a), a)

    a += torch.ones((1, ))     # 原位操作，即in_place操作  在原始地址上直接进行改变
    print(id(a), a)

# flag = True
flag = False
if flag:
    w = torch.tensor([1.], requires_grad=True)   # w = 1
    x = torch.tensor([2.], requires_grad=True)   # x = 2

    a = torch.add(w, x)
    b = torch.add(w, 1)
    y = torch.mul(a, b)

    w.add_(1)

    y.backward()
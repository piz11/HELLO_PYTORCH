# -*- coding: utf-8 -*-
# @File name: lesson-1.py
# @Time : 2022-9-19 11:19
# @Author : zyy
# @Brief : 张量简介与创建

# 创建一维数组
import numpy as np
array_1d = np.ones(3)    # 元素的默认数据类型为float型。
print(array_1d)

# 创建多维数组
# import numpy as np
array_2d = np.ones((2, 3))
print(array_2d)

# 创建int型多维数组
# import numpy as np
array_2d_int = np.ones((2, 3), dtype=int)    # 元素的默认数据类型为float型。
print(array_2d_int)

# 具有元组数据类型和一个的Numpy个数组
# import numpy as np
array_mix_type = np.ones((2, 3), dtype=[('x', 'int'),('y', 'float')])    # 元素的默认数据类型为float型。
print(array_mix_type)
print(array_mix_type.dtype)

print('-------------------------------------------------------------------------')

# 直接创建张量
# import numpy as np
import torch
array = np.ones((3,3))
print("array的数据类型是：", array.dtype)
t1 = torch.tensor(array)
t2 = torch.tensor(array, device='cuda')
print(t1)
print(t2)

# 从numpy创建tensor
import numpy as np
import torch

arr = np.array([[1, 2, 3], [4, 5, 6]])
t = torch.from_numpy(arr)
print(arr)
print(t)

print('\n修改arr')
arr[0, 0] = 0
print(arr)
print(t)

print('\n修改tensor')
t[0, 0] = -1
print(arr)
print(t)

# 依据数值创建tensor
# torch.zeros
# import torch
out_t = torch.tensor([1])
print(out_t)
t = torch.zeros((3, 3), out=out_t)
print(t)
print(out_t)
print(id(t), id(out_t), id(t) == id(out_t))


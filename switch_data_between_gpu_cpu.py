"""
References:
	- https://zhuanlan.zhihu.com/p/31936740
"""
import torch
from torch.autograd import Variable


"""
availability
"""
# whether there is a available GPU
torch.cuda.is_available()
# the amount of available GPU
torch.cuda.device_count()

"""
migration of data
	- Tensor
	- Variable (a "container" of Tensor)

- `.cuda()`默认使用GPU 0
- 对于不同位置的data，是不能直接相互计算的
"""
a = torch.randn(3,4)
# move tensor `a` to GPU
a_gpu = a.gpu()
# move back to cpu
a_cpu = a_gpu.cpu()
# !!!!!!!!!!!!!!!!!!! move Tensor to GPU + convert to Variable != Convert to Variable + move to GPU !!!!!!!!!!!!!
# Since v1 is created before moving to GPU, v1.grad_fn 会记录创建方式
v1_cpu = Variable(a)
v1 = v1_cpu.cuda()
v2 = Variable(a_gpu)


"""
Migration of model
	- moves its weights to GPU
ATTENTION !!!!!!!!!!!!!
	- even if you have used `.cuda()` method, `torch.Tensor()` in your torch.nn.Module still creates non-cuda data!!!
"""
model = model.cuda()
# To see if a model is in GPU, use:
model.weight

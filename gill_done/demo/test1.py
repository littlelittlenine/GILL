import torch

# 假设有包含 tensor 类型张量和元组的列表
data = [(torch.tensor([40, 44, 48]), (52, 56, 60))]

# 对张量部分进行除以4
tensor_result = [tensor / 4 for tensor, _ in data]
print(tensor_result)
# 对元组部分进行除以4
tuple_result = [tuple(value / 4 for value in values) for _, values in data]

# 结合张量和元组部分，得到最终结果
final_result = list(zip(tensor_result, tuple_result))

# 输出最终结果
print(final_result)

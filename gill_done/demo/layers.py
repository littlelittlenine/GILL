import torch
from torch import nn

# 这里是设置了GILLmapper映射：在前后都添加了线性层；与之并列的是linear层，单独的用于检索（这是两个不同的路径）
class TextFcLayer(nn.Module):
  """Layers used in mapping text embeddings to visual outputs. 将文本嵌入映射到视觉输出中使用的层。"""
  # 模式：线性层
  def __init__(self, in_dim: int, out_dim: int, num_input_tokens: int = 1, num_output_tokens: int = 1, mode: str = 'linear'):
    super().__init__()

    self.num_input_tokens = num_input_tokens
    self.num_output_tokens = num_output_tokens
    self.mode = mode
    # 这前面应该有个决策机制呀
    if mode == 'linear':
      self.model = nn.Linear(in_dim, out_dim)
    elif mode == 'gill_mapper':
      hidden_dim = 512
      # 将输入维度为 in_dim 的文本嵌入映射到隐藏维度为 hidden_dim 的空间。这一层的作用是将文本特征映射到一个更高维度的隐藏表示空间
      self.fc = nn.Linear(in_dim, hidden_dim)
      # transformer 模型是一种用于序列到序列 (seq2seq) 学习任务的模型
      # 编码器和解码器的层数
      # Transformer 中前馈神经网络的隐藏层维度
      # 多头注意力机制中注意力头的数量
      self.tfm = nn.Transformer(batch_first=True, norm_first=True,
                                d_model=hidden_dim, num_encoder_layers=4, num_decoder_layers=4,
                                dim_feedforward=hidden_dim * 4, dropout=0.0, nhead=4)
      self.model = nn.Linear(hidden_dim, out_dim)
      # 1 表示这个张量的第一维度大小为 1，这通常用于批处理中的样本数。
      # num_output_tokens 是输出的标记（token）数量，表示查询的标记数量。
      # hidden_dim 是隐藏维度，它决定了每个标记的嵌入维度。
      # 定义了一个可学习的参数 query_embs，它是一个 PyTorch 的可训练参数 (Parameter),query_embs查询向量：通常用于注意力机制中，其中每个标记都有对应的查询向量，用于在计算注意力分布时进行加权求和
      # num_output_tokens是输出的向量个数
      self.query_embs = nn.Parameter(torch.randn(1, num_output_tokens, hidden_dim))
    else:
      raise NotImplementedError(mode)

  def forward(self, x: torch.Tensor, input_embs: torch.Tensor) -> torch.Tensor:
    outputs = None
    
    if self.mode == 'gill_mapper':
      # 张量相应位置的元素值相加，广播机制扩展张量（需要留意调用的时候x，input_embs是什么）
      x = x + input_embs
    # self.model类型是nn.ModuleList（存储模型组件）
    if isinstance(self.model, nn.ModuleList):
      # 模型组件的数量等于输入的向量数？？
      assert len(self.model) == x.shape[1] == self.num_input_tokens, (len(self.model), x.shape, self.num_input_tokens)
      outputs = []
      for i in range(self.num_input_tokens):
        outputs.append(self.model[i](x[:, i, :]))  # (N, D)
      outputs = torch.stack(outputs, dim=1)  # (N, T, D)
    else:
      # 训练的时候是经过下面的代码，不经过上面的
      if self.mode == 'gill_mapper':
        x = self.fc(x)
        # 为什么不需要设置key和value举矩阵
        x = self.tfm(x, self.query_embs.repeat(x.shape[0], 1, 1))
      outputs = self.model(x)
      # 选择单独的linear路径是因为实际输出标记的数量不等于指定的self.num_output_tokens！！！
      if outputs.shape[1] != self.num_output_tokens and self.mode == 'linear':
        if self.mode == 'linear':
          # 重新设置范围
          outputs = outputs[:, :self.num_output_tokens, :]
        else:
          raise NotImplementedError
    
    assert outputs.shape[1] == 1 or (outputs.shape[1] * outputs.shape[2] == self.num_output_tokens * 768), (outputs.shape, self.num_output_tokens)
    return outputs  # (N, T, D)


"""Modified from https://github.com/mlfoundations/open_clip"""

from typing import Optional, Tuple, List

import collections
import logging
import os
import numpy as np
import pandas as pd
import torch
import torchvision.datasets as datasets
from torchvision import transforms as T
from PIL import Image, ImageFont
from torch.utils.data import Dataset

from gill import utils


def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


def get_dataset(args, split: str, tokenizer, precision: str = 'fp32') -> Dataset:
  # split可供选择输入的就是train和val
  assert split in ['train', 'val'
    ], 'Expected split to be one of "train" or "val", got {split} instead.'

  dataset_paths = []
  image_data_dirs = []
  # 如果split==train，train=1
  train = split == 'train'

  # Default configs for datasets.
  # Folder structure should look like: 文件夹结构
  if split == 'train':
    if 'cc3m' in args.dataset:
      # 这里的dataset_dir不存在吗？？？
      dataset_paths.append(os.path.join(args.dataset_dir, 'cc3m_train.tsv'))
      image_data_dirs.append(os.path.join(args.image_dir, 'cc3m/training/'))
    else:
      raise NotImplementedError

  elif split == 'val':
    if 'cc3m' in args.val_dataset:
      # img_dir也不存在吗？？？
      dataset_paths.append(os.path.join(args.dataset_dir, 'cc3m_val.tsv'))
      image_data_dirs.append(os.path.join(args.image_dir, 'cc3m/validation'))
    else:
      raise NotImplementedError
    # 确实里面的路径数为1
    assert len(dataset_paths) == len(image_data_dirs) == 1, (dataset_paths, image_data_dirs)
  else:
    raise NotImplementedError

  if len(dataset_paths) > 1:
    print(f'{len(dataset_paths)} datasets requested: {dataset_paths}')
    dataset = torch.utils.data.ConcatDataset([
      CsvDataset(path, image_dir, tokenizer, 'image',
        'caption', args.visual_model, train=train, max_len=args.max_len, precision=args.precision,
        image_size=args.image_size, retrieval_token_idx=args.retrieval_token_idx, gen_token_idx=args.gen_token_idx, 
        num_tokens=args.num_tokens, num_clip_tokens=args.num_clip_tokens)
      for (path, image_dir) in zip(dataset_paths, image_data_dirs)])
  elif len(dataset_paths) == 1:
    dataset = CsvDataset(dataset_paths[0], image_data_dirs[0], tokenizer, 'image',
      'caption', args.visual_model, train=train, max_len=args.max_len, precision=args.precision,
      image_size=args.image_size, retrieval_token_idx=args.retrieval_token_idx, gen_token_idx=args.gen_token_idx, 
      num_tokens=args.num_tokens, num_clip_tokens=args.num_clip_tokens)
  else:
    raise ValueError(f'There should be at least one valid dataset, got train={args.dataset}, val={args.val_dataset} instead.')
  return dataset

# 定义了一个 CsvDataset 类，该类继承自 PyTorch 的 Dataset 类，用于创建自定义数据集。
# input_filename：CSV 文件的路径。
# base_image_dir：图像文件的基本路径。
# tokenizer：用于文本标记化的分词器。
# img_key：CSV 中包含图像文件名的列的名称。
# caption_key：CSV 中包含图像描述的列的名称。
# feature_extractor_model：特征提取器模型的名称。
# train：指示是否处于训练模式的布尔值，默认为 True。
# max_len：描述文本的最大长度，默认为 32。
# sep：CSV 文件的字段分隔符，默认为制表符 "\t"。
# precision：精度参数，默认为 'fp32'。
# image_size：图像尺寸，默认为 224。
# retrieval_token_idx：检索令牌索引的列表，默认为 [-1]。
# gen_token_idx：生成令牌索引的列表，默认为 [-1]。
# num_tokens：令牌数量，默认为 1。
# num_clip_tokens：CLIP 令牌数量，默认为 1。
class CsvDataset(Dataset):
  def __init__(self, input_filename, base_image_dir, tokenizer, img_key,
               caption_key, feature_extractor_model: str,
               train: bool = True, max_len: int = 32, sep="\t", precision: str = 'fp32',
               image_size: int = 224, retrieval_token_idx: List[int] = [-1], gen_token_idx: List[int] = [-1],
               num_tokens: int = 1, num_clip_tokens: int = 1):
    # 使用 Python logging 模块记录调试信息，提示正在从指定的 TSV 文件中加载数据
    logging.debug(f'Loading tsv data from {input_filename}.')
    # 使用 pandas 库的 read_csv() 函数读取 CSV 文件并将其存储在 DataFrame 中
    df = pd.read_csv(input_filename, sep=sep)

    self.base_image_dir = base_image_dir
    self.images = df[img_key].tolist()
    self.captions = df[caption_key].tolist()
    assert len(self.images) == len(self.captions)

    self.feature_extractor_model = feature_extractor_model
    self.feature_extractor = utils.get_feature_extractor_for_model(
      feature_extractor_model, image_size=image_size, train=False)
    self.image_size = image_size

    self.tokenizer = tokenizer
    self.max_len = max_len
    self.precision = precision
    self.retrieval_token_idx = retrieval_token_idx
    self.gen_token_idx = gen_token_idx
    self.num_tokens = num_tokens
    self.num_clip_tokens = num_clip_tokens

    self.font = None

    logging.debug('Done loading data.')
  # ef __len__(self):：定义了 CsvDataset 类的 __len__ 方法，用于返回数据集的长度（样本数量）
  def __len__(self):
    return len(self.captions)
  # 用于获取数据集中特定索引位置的样本。这个方法接收一个索引 idx 作为参数
  # 返回值是图像路径img_dir； 图像的像素numpy images。
  def __getitem__(self, idx):
    # 在处理样本时会一直执行下去，直到成功获取到一个有效的样本或者出现了异常
    while True:
      # 构建图像文件的完整路径
      image_path = os.path.join(self.base_image_dir, str(self.images[idx]))
      # 获取对应图像描述
      caption = str(self.captions[idx])
      # 构建 CLIP 嵌入文件的路径，按照这里是每一个图片都有一个对应的clip嵌入文件
      clip_l_path = os.path.join(self.base_image_dir, 'clip_embs', str(self.images[idx]) + '.npy')
      # 尝试执行以下代码块，如果出现异常则转到 except 语句块处理。
      try:
        # 打开图像文件
        img = Image.open(image_path)
        # 使用特征提取器提取图像的像素值，并存储在 images 变量中；注意是像素值
        images = utils.get_pixel_values_for_model(self.feature_extractor, img)

        # Only load if we are in generation mode.
        # 使用二进制模式打开 CLIP 嵌入文件
        with open(clip_l_path, 'rb') as f:
          # 从文件中加载 CLIP 嵌入，clip_emb 是一个二维 NumPy 数组，其形状为 (num_clip_tokens, 768)，其中 num_clip_tokens 表示 CLIP 嵌入的数量。
          clip_emb = np.load(f, allow_pickle=True)   # (num_clip_tokens, 768)
          # 选的这么随机吗？？？
          clip_emb = clip_emb[:self.num_clip_tokens, :]

        # Generation mode.
        caption = caption
        # 循环遍历 self.num_tokens 次，self.num_tokens 表示要添加的图像标记的数量。
        for i in range(self.num_tokens):
          # 在图像描述后面添加 [IMG{i}]，其中 {i} 表示当前循环的索引，用于标记图像的位置
          caption += f'[IMG{i}]'
        # 使用分词器对处理后的图像描述进行标记化，得到标记化后的数据
        tokenized_data = self.tokenizer(
          caption,
          return_tensors="pt",
          padding='max_length',
          truncation=True,
          max_length=self.max_len)
        # 获取标记化后的整个序列的ids
        tokens = tokenized_data.input_ids[0]
        # 计算图像描述的长度，也就是第一个维度的和
        caption_len = tokenized_data.attention_mask[0].sum()
        # 注意！！！ 接下来的代码段主要用于处理标记化后的数据，包括替换标记、解码标记等操作。
        # If IMG tokens are overridden by padding, replace them with the correct token.
        # 如果图像标记被填充覆盖，请用正确的标记替换它们
        # 检查标记化后的文本中最后一个标记是否是填充标记或者生成标记中的最后一个标记
        if tokens[-1] not in [self.tokenizer.pad_token_id, self.gen_token_idx[-1]]:
          tokens[-self.num_tokens:] = torch.tensor(self.gen_token_idx).to(dtype=tokens.dtype, device=tokens.device)
        # 使用分词器解码标记化后的文本，生成文本描述。skip_special_tokens=False 确保不跳过特殊标记，例如 [CLS]、[SEP] 等
        # 转化程原本的文本格式
        decode_caption = self.tokenizer.decode(tokens, skip_special_tokens=False)
        # 加载默认字体，如果之前未加载过字体则加载
        self.font = self.font or ImageFont.load_default()
        # 使用解码后的文本描述创建图像，create_image_of_text 函数将文本描述转换为图像
        # 转化程ASCII码的编码形式，但是忽略其他无法编码的元素
        cap_img = utils.create_image_of_text(decode_caption.encode('ascii', 'ignore'), width=self.image_size, nrows=2, font=self.font)
        # 返回值：图像路径；图像像素矩阵；带cap的图像， 
        return image_path, images, cap_img, tokens, caption_len, tokens, caption_len, clip_emb
      except Exception as e:
        print(f'Error reading for {image_path} with caption {caption}: {e}')
        # Pick a new example at random.
        idx = np.random.randint(0, len(self)-1)

# CLIP 嵌入文件中存储的矩阵是由 CLIP 模型生成的图像特征嵌入。这些嵌入是对图像的语义表示，可以捕捉到图像中的内容和语境。
# CLIP 模型通过将图像和文本编码到同一特征空间中，从而实现了跨模态的理解能力，使得可以使用相同的嵌入空间来表示图像和文本。
# CLIP 嵌入文件的矩阵被用于生成与图像相关的文本描述。
# 那这个东西是用在第二个阶段的，没经过决策层的时候，是不是第一阶段也是用的这个
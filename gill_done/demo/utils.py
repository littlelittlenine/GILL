from enum import Enum
import subprocess
import sys
import shutil
import torch
import torch.distributed as dist
from torchvision.transforms import functional as F
from torchvision import transforms as T
from transformers import AutoFeatureExtractor
from PIL import Image, ImageDraw, ImageFont, ImageOps
import random
import requests
from io import BytesIO


def dump_git_status(out_file=sys.stdout, exclude_file_patterns=['*.ipynb', '*.th', '*.sh', '*.txt', '*.json']):
  """Logs git status to stdout."""
  subprocess.call('git rev-parse HEAD', shell=True, stdout=out_file)
  subprocess.call('echo', shell=True, stdout=out_file)
  exclude_string = ''
  subprocess.call('git --no-pager diff -- . {}'.format(exclude_string), shell=True, stdout=out_file)


def get_image_from_url(url: str):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    img = img.resize((224, 224))
    img = img.convert('RGB')
    return img


def truncate_caption(caption: str) -> str:
  """Truncate captions at periods and newlines."""
  caption = caption.strip('\n')
  trunc_index = caption.find('\n') + 1
  if trunc_index <= 0:
      trunc_index = caption.find('.') + 1
  if trunc_index > 0:
    caption = caption[:trunc_index]
  return caption


def pad_to_size(x, size=256):
  delta_w = size - x.size[0]
  delta_h = size - x.size[1]
  padding = (
    delta_w // 2,
    delta_h // 2,
    delta_w - (delta_w // 2),
    delta_h - (delta_h // 2),
  )
  new_im = ImageOps.expand(x, padding)
  return new_im


class RandCropResize(object):

  """
  Randomly crops, then randomly resizes, then randomly crops again, an image. Mirroring the augmentations from https://arxiv.org/abs/2102.12092
  """

  def __init__(self, target_size):
    self.target_size = target_size

  def __call__(self, img):
    img = pad_to_size(img, self.target_size)
    d_min = min(img.size)
    img = T.RandomCrop(size=d_min)(img)
    t_min = min(d_min, round(9 / 8 * self.target_size))
    t_max = min(d_min, round(12 / 8 * self.target_size))
    t = random.randint(t_min, t_max + 1)
    img = T.Resize(t)(img)
    if min(img.size) < 256:
      img = T.Resize(256)(img)
    return T.RandomCrop(size=self.target_size)(img)


class SquarePad(object):
  """Pads image to square.
  From https://discuss.pytorch.org/t/how-to-resize-and-pad-in-a-torchvision-transforms-compose/71850/9
  """
  def __call__(self, image):
    max_wh = max(image.size)
    p_left, p_top = [(max_wh - s) // 2 for s in image.size]
    p_right, p_bottom = [max_wh - (s+pad) for s, pad in zip(image.size, [p_left, p_top])]
    padding = (p_left, p_top, p_right, p_bottom)
    return F.pad(image, padding, 0, 'constant')

# 创建包含文本的图像
# text是要在图像中显示的文本，width是图像的宽度，默认为224像素，nrows是要显示文本的行数，默认为2行，color是文本的颜色，
# 默认为白色，font是要使用的字体，默认为None。函数返回一个PyTorch张量（tensor）。
def create_image_of_text(text: str, width: int = 224, nrows: int = 2, color=(255, 255, 255), font=None) -> torch.Tensor:
  """Creates a (3, nrows * 14, width) image of text. 创建一个（3，nrows * 14，width）文本图像

  Returns:
    cap_img: (3, 14 * nrows, width) image of wrapped text.
  返回：
  cap_img：（3，14 * nrows，width）换行文本的图像
  """
  # 
  # 设置文本行的高度为12像素：在这个函数中，文本行的高度被设定为12像素，这意味着每行文本的高度为12像素，就是字面上的意思，为了在图像上可视化文本
  # 设置文本周围的填充为5像素，文本的边界！！，下面实在
  height = 12
  padding = 5
  # 计算有效宽度，即去除填充后的宽度
  effective_width = width - 2 * padding
  # Create a black image to draw text on.
  # 创建一个黑色背景的图像，大小为有效宽度乘以行数，高度为设定的文本行高度。
  cap_img = Image.new('RGB', (effective_width * nrows, height), color = (0, 0, 0))
  # 创建一个用于在图像上绘制的Draw对象，调用ImageDraw.Draw()函数，并传入要绘制的图像对象（cap_img）
  draw = ImageDraw.Draw(cap_img)
  # 在图像上绘制文本，起始位置为(0, 0)，使用给定的颜色和字体
  draw.text((0, 0), text, color, font=font or ImageFont.load_default())
  # 将 PIL 图像对象 cap_img 转换为 PyTorch 张量，前面的3是视觉里面的通道数。
  cap_img = F.convert_image_dtype(F.pil_to_tensor(cap_img), torch.float32)  # (3, height, W * nrows)
  # 切分，以effective_width为单元水平切分
  cap_img = torch.split(cap_img, effective_width, dim=-1)  # List of nrow elements of shape (3, height, W)
  # 在第一个维度拼接
  cap_img = torch.cat(cap_img, dim=1)  # (3, height * nrows, W)
  # Add zero padding.
  # 在图像的上下左右四个方向上添加对应填充（不在左侧填充），以补充填充区域的大小。
  cap_img = torch.nn.functional.pad(cap_img, [padding, padding, 0, padding])
  return cap_img

# 接受三个参数：model_name（字符串类型）、image_size（整数类型，默认值为224）和train（布尔类型，默认值为True）
# 从transformer导入的AutoFeatureExtractor（特征提取器）
def get_feature_extractor_for_model(model_name: str, image_size: int = 224, train: bool = True):
  print(f'Using HuggingFace AutoFeatureExtractor for {model_name}.')
  # 这个其实就是下载并加载特征提取器
  feature_extractor = AutoFeatureExtractor.from_pretrained("transformer_cache/visual")
  return feature_extractor


def get_pixel_values_for_model(feature_extractor, img: Image.Image):
  pixel_values = feature_extractor(img.convert('RGB'), return_tensors="pt").pixel_values[0, ...]  # (3, H, W)
  return pixel_values


def save_checkpoint(state, is_best, filename='checkpoint'):
  torch.save(state, filename + '.pth.tar')
  if is_best:
    shutil.copyfile(filename + '.pth.tar', filename + '_best.pth.tar')


def accuracy(output, target, padding, topk=(1,)):
  """Computes the accuracy over the k top predictions for the specified values of k"""
  with torch.no_grad():
    maxk = max(topk)
    if output.shape[-1] < maxk:
      print(f"[WARNING] Less than {maxk} predictions available. Using {output.shape[-1]} for topk.")

    maxk = min(maxk, output.shape[-1])
    batch_size = target.size(0)

    # Take topk along the last dimension.
    _, pred = output.topk(maxk, -1, True, True)  # (N, T, topk)

    mask = (target != padding).type(target.dtype)
    target_expand = target[..., None].expand_as(pred)
    correct = pred.eq(target_expand)
    correct = correct * mask[..., None].expand_as(correct)

    res = []
    for k in topk:
      correct_k = correct[..., :k].reshape(-1).float().sum(0, keepdim=True)
      res.append(correct_k.mul_(100.0 / mask.sum()))
    return res

# model模型， max_name_len参数名称的最大长度
def get_params_count(model, max_name_len: int = 60):
  # 参数名称； 参数本身
  # 四个元素组成一个元组存储在params的列表中
  params = [(name[:max_name_len], p.numel(), str(tuple(p.shape)), p.requires_grad) for name, p in model.named_parameters()]
  # 分别计算可训练参数和不可选连参数的总数
  total_trainable_params = sum([x[1] for x in params if x[-1]])
  total_nontrainable_params = sum([x[1] for x in params if not x[-1]])
  return params, total_trainable_params, total_nontrainable_params


def get_params_count_str(model, max_name_len: int = 60):
  padding = 70  # Hardcoded depending on desired amount of padding and separators.
  params, total_trainable_params, total_nontrainable_params = get_params_count(model, max_name_len)
  param_counts_text = ''
  param_counts_text += '=' * (max_name_len + padding) + '\n'
  param_counts_text += f'| {"Module":<{max_name_len}} | {"Trainable":<10} | {"Shape":>15} | {"Param Count":>12} |\n'
  param_counts_text += '-' * (max_name_len + padding) + '\n'
  for name, param_count, shape, trainable in params:
    param_counts_text += f'| {name:<{max_name_len}} | {"True" if trainable else "False":<10} | {shape:>15} | {param_count:>12,} |\n'
  param_counts_text += '-' * (max_name_len + padding) + '\n'
  param_counts_text += f'| {"Total trainable params":<{max_name_len}} | {"":<10} | {"":<15} | {total_trainable_params:>12,} |\n'
  param_counts_text += f'| {"Total non-trainable params":<{max_name_len}} | {"":<10} | {"":<15} | {total_nontrainable_params:>12,} |\n'
  param_counts_text += '=' * (max_name_len + padding) + '\n'
  return param_counts_text


class Summary(Enum):
  NONE = 0
  AVERAGE = 1
  SUM = 2
  COUNT = 3


class ProgressMeter(object):
  def __init__(self, num_batches, meters, prefix=""):
    self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
    self.meters = meters
    self.prefix = prefix

  def display(self, batch):
    entries = [self.prefix + self.batch_fmtstr.format(batch)]
    entries += [str(meter) for meter in self.meters]
    print('\t'.join(entries))
    
  def display_summary(self):
    entries = [" *"]
    entries += [meter.summary() for meter in self.meters]
    print(' '.join(entries))

  def _get_batch_fmtstr(self, num_batches):
    num_digits = len(str(num_batches // 1))
    fmt = '{:' + str(num_digits) + 'd}'
    return '[' + fmt + '/' + fmt.format(num_batches) + ']'


class AverageMeter(object):
  """Computes and stores the average and current value"""
  def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
    self.name = name
    self.fmt = fmt
    self.summary_type = summary_type
    self.reset()

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count

  def all_reduce(self):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
    dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
    self.sum, self.count = total.tolist()
    self.avg = self.sum / self.count

  def __str__(self):
    fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
    return fmtstr.format(**self.__dict__)
  
  def summary(self):
    fmtstr = ''
    if self.summary_type is Summary.NONE:
      fmtstr = ''
    elif self.summary_type is Summary.AVERAGE:
      fmtstr = '{name} {avg:.3f}'
    elif self.summary_type is Summary.SUM:
      fmtstr = '{name} {sum:.3f}'
    elif self.summary_type is Summary.COUNT:
      fmtstr = '{name} {count:.3f}'
    else:
      raise ValueError('invalid summary type %r' % self.summary_type)
    
    return fmtstr.format(**self.__dict__)

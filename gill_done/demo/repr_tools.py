"""
Contains utilities for extracting token representations and indices
from string templates. Used in computing the left and right vectors for ROME.

包含从字符串模板中提取令牌表示和索引的实用工具。用于计算 ROME 的左向量和右向量
"""

from copy import deepcopy
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from util import nethook

# 专门提取某一个word某一个模块的输入和输出，要提取的word是不是就是subtoken
# sub_token的含义
def get_reprs_at_word_tokens(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    context_templates: List[str],
    words: List[str],
    layer: int,
    module_template: str,
    subtoken: str,
    track: str = "in",
) -> torch.Tensor:
    """
    Retrieves the last token representation of `word` in `context_template`
    when `word` is substituted into `context_template`. See `get_last_word_idx_in_template`
    for more details.
    当将 word 替换到 context_template 中时，检索 context_template 中 word 的最后一个token表示
    """
   
    idxs = get_words_idxs_in_templates(tok, context_templates, words, subtoken)
    return get_reprs_at_idxs(
        model,
        tok,
        # 将对应的word填入对应的context_templates中
        [context_templates[i].format(words[i]) for i in range(len(words))],
        idxs,
        layer,
        module_template,
        track,
    )

# 在给定模板字符串列表、带有一个格式说明符的模板字符串以及要替换到模板中的单词的情况下，计算它们最后一个令牌的标记化索引
# 接收四个输入参数
# tok（AutoTokenizer 类型的对象）
# context_templates（字符串列表）
# words（字符串列表），单词的列表
# subtoken（字符串类型）
# context_templates有对应需要填充的位置，而words里面存储的单词则是要填充进去
def get_words_idxs_in_templates(
    tok: AutoTokenizer, context_templates: str, words: str, subtoken: str
) -> int:
    """
    Given list of template strings, each with *one* format specifier
    (e.g. "{} plays basketball"), and words to be substituted into the
    template, computes the post-tokenization index of their last tokens.
    给定一个模板字符串列表，每个模板字符串都有 一个 格式说明符（例如，"{} 打篮球"），以及要替换到模板中的单词，计算它们最后一个令牌的标记化索引
    """
    # 检查每个模板字符串中是否有且只有一个格式说明符{}，如果不是则引发 AssertionError，并给出错误消息。
    assert all(
        tmp.count("{}") == 1 for tmp in context_templates
    ), "We currently do not support multiple fill-ins for context"

    # Compute prefixes and suffixes of the tokenized context
    # 计算标记化上下文的前缀和后缀
    # 计算每个模板字符串中格式说明符{}的索引位置，并存储在 fill_idxs 中
    fill_idxs = [tmp.index("{}") for tmp in context_templates]
    # 根据索引位置将每个模板字符串拆分为前缀和后缀部分，并存储在 prefixes 和 suffixes 中，这也是一个列表；因为context_templates是一个列表，所以结果也是对每个模板字符串确定前缀和后缀
    prefixes, suffixes = [
        tmp[: fill_idxs[i]] for i, tmp in enumerate(context_templates)
    ], [tmp[fill_idxs[i] + 2 :] for i, tmp in enumerate(context_templates)]
    # 深度复制 words 列表，以免影响原始数据
    words = deepcopy(words)

    # Pre-process tokens
    # 索引， 前缀内容
    for i, prefix in enumerate(prefixes):
        if len(prefix) > 0:
            # 确保前缀的最后一个字符是空格，应该是格式化符号{}和前后缀之间都有空格
            assert prefix[-1] == " "
            # 去除前缀末尾的空格
            prefix = prefix[:-1]
            # 更新列表中的前缀
            prefixes[i] = prefix
            # 对单词进行处理，去除前后空格，并在前面添加一个空格，前面添加一个空格是为了拼接。后缀就没有去除前面空格的操作
            words[i] = f" {words[i].strip()}"

    # Tokenize to determine lengths
    # 确保前缀、单词和后缀的数量相等
    assert len(prefixes) == len(words) == len(suffixes)
    n = len(prefixes)
    # 将前缀、单词和后缀合并成一个批量的字符串，然后通过 tok 对象进行标记化处理，得到标记化后的列表
    # 捷豹操作，合成一个新的列表
    # 将列表中的内容转化为token
    batch_tok = tok([*prefixes, *words, *suffixes])
    # 将标记化后的列表拆分成前缀、单词和后缀部分，经过标记化后的token
    prefixes_tok, words_tok, suffixes_tok = [
        batch_tok[i : i + n] for i in range(0, n * 3, n)
    ]
    # 三个列表，比如prefixes_len，存储的是所有前缀的长度（前缀列表中每个单元的长度，单位是token，经过分词后的）
    prefixes_len, words_len, suffixes_len = [
        [len(el) for el in tok_list]
        for tok_list in [prefixes_tok, words_tok, suffixes_tok]
    ]
    
    # Compute indices of last tokens
    # 计算最后一个标记的索引
    # 检查 subtoken 是否为 "last" 或 "first_after_last"
    # 要么是后缀，要么后缀不存在。
    if subtoken == "last" or subtoken == "first_after_last":
        return [
            [
                prefixes_len[i]
                + words_len[i]
                - (1 if subtoken == "last" or suffixes_len[i] == 0 else 0)
            ]
            # If suffix is empty, there is no "first token after the last".
            # So, just return the last token of the word.
            for i in range(n)
        ]
    # 如果 subtoken 是 "first"，则返回每个单词第一个令牌的索引。-
    elif subtoken == "first":
        return [[prefixes_len[i]] for i in range(n)]
    else:
        raise ValueError(f"Unknown subtoken type: {subtoken}")


def get_reprs_at_idxs(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    contexts: List[str],
    idxs: List[List[int]],
    layer: int,
    module_template: str,
    track: str = "in",
) -> torch.Tensor:
    """
    Runs input through model and returns averaged representations of the tokens
    at each index in `idxs`.
    将输入数据输入模型，并返回在 idxs 中每个索引处的令牌的平均表示。

    """
    
    # _batch 的函数，用于生成批处理的上下文和索引。这个函数接受一个参数 n，表示批处理的大小
    def _batch(n):
        for i in range(0, len(contexts), n):
            # 后面得到的batch_idxs就是填入的word的最后一个token
            yield contexts[i : i + n], idxs[i : i + n]

    assert track in {"in", "out", "both"}
    # track==both的话， both设置为1
    both = track == "both"
    # track为in或者both时，tin=true； track为out或者both时， tout=true
    tin, tout = (
        (track == "in" or both),
        (track == "out" or both),
    )
    # module_template.format(layer) 中的 .format(layer) 是 Python 中的字符串格式化方法。在这里，module_template 是一个字符串模板，其中可能包含一个占位符 {}，用于插入后面的参数 layer 的值
    module_name = module_template.format(layer)
    to_return = {"in": [], "out": []}
    # _process 的函数，用于处理当前表示、批处理的索引和键。在循环中，将处理后的表示按照索引计算平均值，并将结果添加到 to_return 字典中对应键值的列表中。
    # 当前表示 (cur_repr)，批处理的索引 (batch_idxs) 和键 (key)，然后将处理后的结果添加到 to_return 字典中对应键值的列表中
    def _process(cur_repr, batch_idxs, key):
        # 声明 to_return 是外部变量，这样在函数内部可以修改外部作用域的 to_return 变量
        nonlocal to_return
        # 如果 cur_repr 是一个元组，则取其中的第一个元素作为当前表示。
        cur_repr = cur_repr[0] if type(cur_repr) is tuple else cur_repr
        # i 是索引的序号，idx_list 是具体的索引列表
        for i, idx_list in enumerate(batch_idxs):
            # .mean(0) 是对一个张量（tensor）进行求平均值操作的方法，其中参数 0 表示在张量的第一个维度上进行求平均值操作。其实比如一个（3，4，5）dim=0求平均得到（4，5）
            to_return[key].append(cur_repr[i][idx_list].mean(0))
    #  _batch 函数生成批处理的上下文和索引，对上下文进行分词操作
    for batch_contexts, batch_idxs in _batch(n=128):
        contexts_tok = tok(batch_contexts, padding=True, return_tensors="pt").to(
            next(model.parameters()).device
        )
        # 创建一个上下文管理器 nethook.Trace，用于跟踪模型的输入和输出。在 with 块内，运行模型传入上下文张量，并记录输入和输出。
        # 输入和输出数据会被记录在 tr 中，输入和输出是自动记录的吧。可以直接调用属性
        with torch.no_grad():
            with nethook.Trace(
                module=model,
                layer=module_name,
                retain_input=tin,
                retain_output=tout,
            ) as tr:
                model(**contexts_tok)

        if tin:
            _process(tr.input, batch_idxs, "in")
        if tout:
            _process(tr.output, batch_idxs, "out")
    # 处理只关注输入或者输出时，有一个列表为0
    to_return = {k: torch.stack(v, 0) for k, v in to_return.items() if len(v) > 0}
    
    if len(to_return) == 1:
        return to_return["in"] if tin else to_return["out"]
    else:
        return to_return["in"], to_return["out"]

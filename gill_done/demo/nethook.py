"""
Utilities for instrumenting a torch model. 用于为 torch 模型进行仪器化的工具。

Trace will hook one layer at a time.      Trace 将一次钩住一层。
TraceDict will hook multiple layers at once.  TraceDict一次钩住多个层。
subsequence slices intervals from Sequential modules.   从 Sequential 模块中切片子序列间隔
get_module, replace_module, get_parameter resolve dotted names.  get_module、replace_module、get_parameter 解析点分隔的名称。
set_requires_grad recursively sets requires_grad in module parameters.  set_requires_grad 递归设置模块参数中的 requires_grad。
"""

import contextlib
import copy
import inspect
from collections import OrderedDict

import torch

# Trace的类，用于在计算神经网络时保留指定层的输出
class Trace(contextlib.AbstractContextManager):
    """
    为了在计算给定网络时保留指定层的输出
    To retain the output of the named layer during the computation of
    the given network:

        with Trace(net, 'layer.name') as ret:
            _ = net(inp)
            representation = ret.output

    A layer module can be passed directly without a layer name, and 可以直接传递一个层模块而不需要层名称，其输出将会被保留。默认情况下，会返回对输出对象的直接引用，但可以通过选项来控制
    its output will be retained.  By default, a direct reference to
    the output object is returned, but options can control this:

        clone=True  - retains a copy of the output, which can be    clone=True - 保留输出的副本，如果您希望在稍后网络可能就地修改输出之前查看输出，则这可能很有用
            useful if you want to see the output before it might
            be modified by the network in-place later.
        detach=True - retains a detached reference or copy.  (By    detach=True - 保留分离的引用或副本。（默认情况下，该值将保持连接到图形。）
            default the value would be left attached to the graph.)
        retain_grad=True - request gradient to be retained on the
            output.  After backward(), ret.output.grad is populated.

        retain_input=True - also retains the input.                 retain_grad=True - 请求在输出上保留梯度。在调用backward()后，ret.output.grad会被填充。
        retain_output=False - can disable retaining the output.     retain_input=True - 也保留输入
        edit_output=fn - calls the function to modify the output    retain_output=False - 可以禁用保留输出
            of the layer before passing it the rest of the model.   edit_output=fn - 在将输出传递给模型的其余部分之前调用该函数以修改层的输出。fn可以选择接受（原始输出，层名称）参数
            fn can optionally accept (output, layer) arguments      stop=True - 在运行该层后抛出StopForward异常，这允许仅运行模型的一部分   
            for the original output and the layer name.            
        stop=True - throws a StopForward exception after the layer  感觉
            is run, which allows running just a portion of a model.
    """

    def __init__(
        self, 
        # 要追踪的神经网络模型                    
        module,
        # 要追踪输出的层的名称，可以是空值，直接传递一个层模块
        layer=None,
        # 控制是否保留输出，默认为True。
        retain_output=True,
        # 控制是否保留输入，默认为False
        retain_input=False,
        # 控制是否保留输出的副本，默认为False
        clone=False,
        # 控制是否保留分离的引用或副本，默认为False
        detach=False,
        # 控制是否保留梯度，默认为False
        retain_grad=False,
        # 处理修改层输出的函数，默认为None。该函数可以选择接受(output, layer)参数，用于处理原始输出和层名称
        edit_output=None,
        # 控制是否在运行该层后抛出StopForward异常，允许仅运行模型的一部分，默认为False
        stop=False,
    ):
        """
        Method to replace a forward method with a closure that
        intercepts the call, and tracks the hook so that it can be reverted.
        一种用闭包替换前向方法的方法，该闭包拦截调用，并跟踪挂钩以便可以恢复
        闭包：闭包（closure）是指一个函数与其相关的引用环境的组合。闭包允许函数访问其自身定义范围外的变量，即使创建闭包的上下文已经不存在
        前向方法： 将操作或请求传递给另一个对象或组件的方法
        闭包拦截调用，并跟踪挂钩以便可以恢复这句话的意思是：
        闭包在实现时会拦截函数的调用，并记录相关的信息，例如执行的时间点、传入的参数等。这些记录被称为“挂钩”（hooks），可以用来在之后恢复函数的执行状态，重新执行相同的操作。
        闭包通过捕获和保存这些信息，可以在需要时重新创建函数的上下文，以便继续或重新执行特定的操作。
        """
        # 类实例self赋值给变量retainer
        # 这样做通常是为了在类的方法中可以访问类的属性和方法， 通过retainer
        retainer = self
        # 要跟踪的层，默认为空
        self.layer = layer
        if layer is not None:
            # 这是为了获取追踪的特定层的模块对象
            # 在module中查找指定名称layer的模块
            module = get_module(module, layer)
        # m是神经网络模块； inputs是传入该模块的输入数据， output是该模块的输出数据????输出数据也作为输入参数吗
            
        # retain_hook 的函数，该函数被用作注册到神经网络模块的前向传播钩子
        # 函数主要是用于在神经网络模块的前向传播过程中进行指定的操作，如保存输入、处理输出、复制数据等。前向传播钩子是一种强大的工具，可以在神经网络运行过程中注入自定义的逻辑，对模型的行为进行调整和控制。
        # 钩子函数的实现内容：对输入的深度复刻; 修改输出层; 对输出进行复刻;  
        # output的输出和retain_output， retain.grad有关
        def retain_hook(m, inputs, output):
            # 实现对输入的深度复刻
            if retain_input:
                retainer.input = recursive_copy( 
                    inputs[0] if len(inputs) == 1 else inputs,
                    clone=clone,
                    detach=detach,
                    retain_grad=False,
                )  # retain_grad applies to output only. 梯度信息只适用于输出，而不是输入
            if edit_output:
                # 调用自定义的edit_output函数
                output = invoke_with_optional_args(
                    edit_output, output=output, layer=self.layer
                )
            # 对输出数据进行深度复刻。
            if retain_output:
                retainer.output = recursive_copy(
                    output, clone=clone, detach=detach, retain_grad=retain_grad
                )
                # When retain_grad is set, also insert a trivial
                # copy operation.  That allows in-place operations
                # to follow without error.
                if retain_grad:
                    output = recursive_copy(retainer.output, clone=True, detach=False)
            if stop:
                raise StopForward()
            return output
        #retain_hook 函数注册为神经网络模块的前向传播钩子，以便在每次前向传播时执行相关操作
        # 太对了，在前向传播过程中调用钩子函数
        self.registered_hook = module.register_forward_hook(retain_hook)
        self.stop = stop

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()
        if self.stop and issubclass(type, StopForward):
            return True

    def close(self):
        self.registered_hook.remove()
class Trace_1(contextlib.AbstractContextManager):
    def __init__(
        self, 
        # 要追踪的神经网络模型                    
        module=None,
        # 要追踪输出的层的名称，可以是空值，直接传递一个层模块
        layer=None,
        # 控制是否保留输出，默认为True。
        retain_output=True,
        # 控制是否保留输入，默认为False
        retain_input=False,
        # 控制是否保留输出的副本，默认为False
        clone=False,
        # 控制是否保留分离的引用或副本，默认为False
        detach=False,
        # 控制是否保留梯度，默认为False
        retain_grad=False,
        # 处理修改层输出的函数，默认为None。该函数可以选择接受(output, layer)参数，用于处理原始输出和层名称
        edit_output=None,
        # 控制是否在运行该层后抛出StopForward异常，允许仅运行模型的一部分，默认为False
        stop=False,
        hand_act = None
    ):
        retainer = self
        # 要跟踪的层，默认为空
        self.layer = layer
        self.act = []
        if module is not None:
            print("no")
        if layer is not None:
            # 这是为了获取追踪的特定层的模块对象
            # 在module中查找指定名称layer的模块
            module = get_module(module, layer)
        # m是神经网络模块； inputs是传入该模块的输入数据， output是该模块的输出数据????输出数据也作为输入参数吗
            
        # retain_hook 的函数，该函数被用作注册到神经网络模块的前向传播钩子
        # 函数主要是用于在神经网络模块的前向传播过程中进行指定的操作，如保存输入、处理输出、复制数据等。前向传播钩子是一种强大的工具，可以在神经网络运行过程中注入自定义的逻辑，对模型的行为进行调整和控制。
        # 钩子函数的实现内容：对输入的深度复刻; 修改输出层; 对输出进行复刻;  
        # output的输出和retain_output， retain.grad有关
        def retain_hook(m, inputs, output):
            # 实现对输入的深度复刻
            if retain_input:
                retainer.input = recursive_copy( 
                    inputs[0] if len(inputs) == 1 else inputs,
                    clone=clone,
                    detach=detach,
                    retain_grad=False,
                )  # retain_grad applies to output only. 梯度信息只适用于输出，而不是输入
            if edit_output:
                # 调用自定义的edit_output函数
                output = invoke_with_optional_args(
                    edit_output, output=output, layer=self.layer
                )
            # 对输出数据进行深度复刻。
            if retain_output:
                retainer.output = recursive_copy(
                    output, clone=clone, detach=detach, retain_grad=retain_grad
                )
                # When retain_grad is set, also insert a trivial
                # copy operation.  That allows in-place operations
                # to follow without error.
                if retain_grad:
                    output = recursive_copy(retainer.output, clone=True, detach=False)
            if stop:
                raise StopForward()
            self.act.append(output)
            return output

        #retain_hook 函数注册为神经网络模块的前向传播钩子，以便在每次前向传播时执行相关操作
        # 太对了，在前向传播过程中调用钩子函数
        # self.registered_hook = module.register_forward_hook(retain_hook)
        self.hand_act = [module.model.lm.model.decoder.layers[n].activation_fn.register_forward_hook(retain_hook) for n in
                           range(32)]
        #self.hand_act = [module.model.lm.model.decoder.layers[0].activation_fn.register_forward_hook(retain_hook)]
        self.stop = stop

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()
        if self.stop and issubclass(type, StopForward):
            return True

    def close(self):
        # self.registered_hook.remove()
        for h in self.hand_act:
            h.remove()

# TraceDict 的类，该类继承自 OrderedDict 并实现了 contextlib.AbstractContextManager，主要用于在计算网络时保留多个命名层的输出，以便后续使用。
# 命名层：在神经网络中有特定名称的层。这里的特定名称就是输入的layers。与Trace对比：Trace处理单个layer；TraceDict调用Trace处理多个。
class TraceDict(OrderedDict, contextlib.AbstractContextManager):
    """
    To retain the output of multiple named layers during the computation
    of the given network:

        with TraceDict(net, ['layer1.name1', 'layer2.name2']) as ret:
            _ = net(inp)
            representation = ret['layer1.name1'].output

    If edit_output is provided, it should be a function that takes
    two arguments: output, and the layer name; and then it returns the
    modified output.

    Other arguments are the same as Trace.  If stop is True, then the
    execution of the network will be stopped after the last layer
    listed (even if it would not have been the last to be executed).
    """

    def __init__(
        self,
        # 神经网络模型或模块，用于计算网络输出
        module,
        # 需要保留输出的层名称列表。
        layers=None,
        # 是否保留输出，默认为 True
        retain_output=True,
        # 是否保留输入，默认为 False
        retain_input=False,
        # 是否克隆输出，默认为 False
        clone=False,
        # 是否断开梯度，默认为 False
        detach=False,
        # 是否保留梯度，默认为 False
        retain_grad=False,
        # 一个函数，用于编辑输出，接受两个参数（输出和层名称），返回修改后的输出， 我感觉是经过边集输出后得到的结果
        edit_output=None,
        # 是否在最后一个指定的层停止网络执行，默认为 False
        stop=False,
    ):
        self.stop = stop
        # 生成器函数，其作用是标记在迭代过程中最后一个未见过的元素
        # 最后会生成一个由二元组组成的列表， 最后的元素都是（True， prev）
        def flag_last_unseen(it):
            try:
                # 将传入的可迭代对象转换为迭代器，以便进行迭代操作
                it = iter(it)
                # 使用 next 函数获取第一个元素作为上一个元素，如果迭代器为空则会引发 StopIteration 异常
                prev = next(it)
                # 创建一个集合 seen，并将第一个元素加入其中，表示已经遇到过的元素
                seen = set([prev])
            except StopIteration:
                return
            # 遍历迭代器中的每个元素
            for item in it:
                # 不在seen中
                if item not in seen:
                    # yield生成一个二元组（False，prev），其中 False 表示当前元素未见过，prev 是上一个元素
                    yield False, prev
                    # 将item加入seen集合中，表示已经见过
                    seen.add(item)
                    # 更新 prev 为当前元素，用于下一次构建二元组
                    prev = item
            yield True, prev
        # layers：需要保留输出的层的列表。 is_last:是否是最后一个元素
        for is_last, layer in flag_last_unseen(layers):
            # 将layers都设置成Trace类：为了计算神经网络时保留指定层的输出。
            self[layer] = Trace(
                module=module,
                layer=layer,
                retain_output=retain_output,
                retain_input=retain_input,
                clone=clone,
                detach=detach,
                retain_grad=retain_grad,
                edit_output=edit_output,
                stop=stop and is_last,
            )

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()
        if self.stop and issubclass(type, StopForward):
            return True

    def close(self):
        for layer, trace in reversed(self.items()):
            trace.close()


class StopForward(Exception):
    """
    If the only output needed from running a network is the retained
    submodule then Trace(submodule, stop=True) will stop execution
    immediately after the retained submodule by raising the StopForward()
    exception.  When Trace is used as context manager, it catches that
    exception and can be used as follows:

    with Trace(net, layername, stop=True) as tr:
        net(inp) # Only runs the network up to layername
    print(tr.output)
    """
    """
    如果在运行网络时唯一需要的输出是保留的子模块，那么通过使用 Trace(submodule, stop=True) 可以立即停止执行，并抛出 StopForward() 异常。当 Trace 作为上下文管理器使用时，它会捕获该异常，并可以按照以下方式使用：
    python
    with Trace(net, layername, stop=True) as tr:
        net(inp)  # 仅运行网络直到 layername
        print(tr.output)
    在这段代码中，net 是神经网络模型，inp 是输入数据。通过 Trace 的上下文管理器，可以控制网络执行到指定的 layername 层后立即停止，并且通过 tr.output 可以获取到停止后的输出结果。
    """
    pass

# 这个函数是上面提到的深度复刻：复刻what

def recursive_copy(x, clone=None, detach=None, retain_grad=None):
    """
    Copies a reference to a tensor, or an object that contains tensors,
    optionally detaching and cloning the tensor(s).  If retain_grad is
    true, the original tensors are marked to have grads retained.
    """
    """
    将一个张量或包含张量的对象的引用复制一份，可以选择性地对张量进行分离和克隆操作。如果 `retain_grad` 参数为 true，则会标记原始张量以保留梯度信息。
    输入的x就是上面提到的张量或柏寒张量的对象的引用
    """
    # clone、detach、retain_grad：这三个参数用于控制复制行为，分别表示是否克隆、是否分离梯度、是否保留梯度
    if not clone and not detach and not retain_grad:
        return x
    if isinstance(x, torch.Tensor):
        # retain_grad 为真，则会标记原始张量以保留梯度信息
        if retain_grad:
            if not x.requires_grad:
                # 确保张量 x 开启了梯度跟踪功能，即需要计算梯度
                x.requires_grad = True
            # 调用 x.retain_grad() 方法，这个方法的作用是标记张量 x 的梯度值需要被保留
            # 如果不调用 retain_grad() 方法，张量在反向传播计算完成后，其梯度值会被释放，不再保留。通过调用 retain_grad() 方法，可以确保在反向传播过程中保留张量的梯度值，以便后续进行梯度优化更新等操作。
            # 为什么需要保留呢？在反向传播过程中保留梯度值
            x.retain_grad()
        elif detach:
            # 对张量进行分离操作，本质上得到的这个x是
            x = x.detach()
        if clone:
            # 对张量进行clone操作
            x = x.clone()
        return x
    # Only dicts, lists, and tuples (and subclasses) can be copied.
    # 对里面的元素进行复制操作
    if isinstance(x, dict):
        return type(x)({k: recursive_copy(v) for k, v in x.items()})
    elif isinstance(x, (list, tuple)):
        return type(x)([recursive_copy(v) for v in x])
    else:
        assert False, f"Unknown type {type(x)} cannot be broken into tensors."

# 用于创建一个 PyTorch Sequential 模型的子序列，可以选择性地复制模块和参数到子序列中
# PyTorch 中的 Sequential 模型是一种简单的容器，用于按顺序组织神经网络的各个层。
# Sequential 模型允许用户按照顺序将各种神经网络层组合在一起，以构建神经网络模型。这种模型的结构非常直观和简单，适用于线性堆叠的层结构，其中每一层都有一个输入和一个输出。
# OK， 那么首先first_layer到last_layer这几个层时平级的。
# sequential包含的层数可能是大于要提取的子模型的
def subsequence(
    # PyTorch Sequential 模型的输入
    sequential,
    # 指定子序列的起始层（包含在子序列中）
    first_layer=None,
    # 指定子序列的结束层（包含在子序列中）
    last_layer=None,
    # 指定子序列的起始层（不包含在子序列中）
    after_layer=None,
    # 指定子序列的结束层（不包含在子序列中）
    upto_layer=None,
    # 用于指定单个层，若指定了该参数，则会忽略其他范围参数，这个单个层将被视为子序列的起始和结束层
    single_layer=None,
    # 一个布尔值，表示是否共享参数
    share_weights=False,
):
    """
    Creates a subsequence of a pytorch Sequential model, copying over
    modules together with parameters for the subsequence.  Only
    modules from first_layer to last_layer (inclusive) are included,
    or modules between after_layer and upto_layer (exclusive).
    Handles descent into dotted layer names as long as all references
    are within nested Sequential models.

    If share_weights is True, then references the original modules
    and their parameters without copying them.  Otherwise, by default,
    makes a separate brand-new copy.
    """
    # 嵌套的Sequential 模型，可以通过点操作下降到内部层
    assert (single_layer is None) or (
        first_layer is last_layer is after_layer is upto_layer is None
    )
    # Sequential 序列里面只有单个层，也就是单个神经网络
    if single_layer is not None:
        first_layer = single_layer
        last_layer = single_layer
    # 这里的first， last， after， upto是为了将输入的层级分开，组成一个列表
    first, last, after, upto = [
        None if d is None else d.split(".")
        for d in [first_layer, last_layer, after_layer, upto_layer]
    ]
    # 上面时为了下面return返回值处理数据，处理数据的操作就是得到起始和结束层的列表
    return hierarchical_subsequence(
        sequential,
        first=first,
        last=last,
        after=after,
        upto=upto,
        share_weights=share_weights,
    )


def hierarchical_subsequence(
    sequential, first, last, after, upto, share_weights=False, depth=0
):
    """
    Recursive helper for subsequence() to support descent into dotted
    layer names.  In this helper, first, last, after, and upto are
    arrays of names resulting from splitting on dots.  Can only
    descend into nested Sequentials.
    """
    """
    递归辅助函数用于支持进入带有点分隔层名称的子序列（subsequence()）。在这个辅助函数中，first、last、after 和 upto 是根据点分隔拆分而成的名称数组。只能进入嵌套的 Sequentials。
    """
    # last 和 upto 以及first 和 after这两组每组只能有一个为None，且他们都是列表，存储了
    assert (last is None) or (upto is None)
    assert (first is None) or (after is None)
    # 都为None； 如果share_weigths=True，那么返回原始的sequential模型，否则深度拷贝原始的sequential模型
    if first is last is after is upto is None:
        return sequential if share_weights else copy.deepcopy(sequential)
    # 确保传入的 sequential 参数是 torch.nn.Sequential 类型的实例
    assert isinstance(sequential, torch.nn.Sequential), (
        ".".join((first or last or after or upto)[:depth] or "arg") + " not Sequential"
    )
    # 都是None的时候including_children为True，表示不包含子模型
    including_children = (first is None) and (after is None)
    # 创建了一个空的有序字典 included_children，用于存储已包括的子模型。
    included_children = OrderedDict()
    # A = current level short name of A. 当前级别的简短名称
    # AN = full name for recursive descent if not innermost. 如果不是最内层，则为递归下降的完整名称
    # 这是一个多重赋值语句，将处理后的名称结果分别赋值给 (F, FN)、(L, LN)、(A, AN)、(U, UN) 这四对变量
    # d是传入的名称数组中的名称。
    # 根据depth的值提取当前层级的深度和完整的名称
    (F, FN), (L, LN), (A, AN), (U, UN) = [
        # d[depth]是当前层的名称， 如果层级列表不为1，FN等要存储列表
        (d[depth], (None if len(d) == depth + 1 else d))
        if d is not None
        else (None, None)
        for d in [first, last, after, upto]
    ]
    # 遍历模型 sequential 的每个层，获取层的名称和对应的模块
    for name, layer in sequential._modules.items():
        # 则将 first 设为 None，表示当前层是需要包含的起始层，同时设置 including_children 为 True，表示需要包括子模型
        if name == F:
            first = None
            including_children = True
        # 表示当前层与 A 对应且不是叶子层，类似于处理 F，将 after 设为 None，同时设置 including_children 为 True
        if name == A and AN is not None:  # just like F if not a leaf.
            after = None
            including_children = True
        # 如果当前层的名称匹配 U 且 UN 为 None，表示当前层是需要到达的最高层，将 upto 设为 None，同时设置 including_children 为 False，表示不包括子模型
        if name == U and UN is None:
            upto = None
            including_children = False
        # 需要包含子模型时
        if including_children: 
            # AR = full name for recursive descent if name matches.
            # AR = 如果名称匹配，则递归下降的完整名称。
            FR, LR, AR, UR = [
                n if n is None or n[depth] == name else None for n in [FN, LN, AN, UN]
            ]
            # 这么递归下去的话，只是F,L,A,U存储的是下降的名称，
            chosen = hierarchical_subsequence(
                layer,
                first=FR,
                last=LR,
                after=AR,
                upto=UR,
                share_weights=share_weights,
                depth=depth + 1,
            )
            if chosen is not None:
                included_children[name] = chosen
        if name == L:
            last = None
            including_children = False
        if name == U and UN is not None:  # just like L if not a leaf.
            upto = None
            including_children = False
        if name == A and AN is None:
            after = None
            including_children = True
    for name in [first, last, after, upto]:
        if name is not None:
            raise ValueError("Layer %s not found" % ".".join(name))
    # Omit empty subsequences except at the outermost level,
    # where we should not return None.
    if not len(included_children) and depth > 0:
        return None
    result = torch.nn.Sequential(included_children)
    result.training = sequential.training
    return result

# 为传入的模型中的所有参数设置 requires_grad 为 true 或 false。
def set_requires_grad(requires_grad, *models):
    """
    Sets requires_grad true or false for all parameters within the
    models passed.
    """
    for model in models:
        if isinstance(model, torch.nn.Module):
            for param in model.parameters():
                param.requires_grad = requires_grad
        elif isinstance(model, (torch.nn.Parameter, torch.Tensor)):
            model.requires_grad = requires_grad
        else:
            assert False, "unknown type %r" % type(model)

# 在给定的模型中查找指定名称的模块
def get_module(model, name):
    """
    Finds the named module within the given model.
    """
    for n, m in model.named_modules():
        if n == name:
            return m
    raise LookupError(name)
def get_module(model, name):
    return model
# 在给定的模型中查找指定名称的参数
def get_parameter(model, name):
    """
    Finds the named parameter within the given model.
    """
    for n, p in model.named_parameters():
        if n == name:
            return p
    raise LookupError(name)

# 替换给定模型中指定名称的模块
def replace_module(model, name, new_module):
    """
    Replaces the named module within the given model.
    """
    if "." in name:
        parent_name, attr_name = name.rsplit(".", 1)
        model = get_module(model, parent_name)
    # original_module = getattr(model, attr_name)
    setattr(model, attr_name, new_module)

# 以一种灵活的方式调用其他函数
def invoke_with_optional_args(fn, *args, **kwargs):
    """
    Invokes a function with only the arguments that it
    is written to accept, giving priority to arguments
    that match by-name, using the following rules.
    (1) arguments with matching names are passed by name.
    (2) remaining non-name-matched args are passed by order.
    (3) extra caller arguments that the function cannot
        accept are not passed.
    (4) extra required function arguments that the caller
        cannot provide cause a TypeError to be raised.
    Ordinary python calling conventions are helpful for
    supporting a function that might be revised to accept
    extra arguments in a newer version, without requiring the
    caller to pass those new arguments.  This function helps
    support function callers that might be revised to supply
    extra arguments, without requiring the callee to accept
    those new arguments.
    """
    # 通过 inspect.getfullargspec(fn) 获取到函数 fn 的参数信息，包括参数名、默认值
    # 使得函数调用在未来的修改过程中更加灵活，不需要修改调用方和被调用函数的参数传递方式
    argspec = inspect.getfullargspec(fn)
    # 函数定义的参数规则和传入的参数，构建一个参数列表 pass_args，用于调用函数 fn
    pass_args = []
    used_kw = set()
    unmatched_pos = []
    used_pos = 0
    defaulted_pos = len(argspec.args) - (
        0 if not argspec.defaults else len(argspec.defaults)
    )
    # Pass positional args that match name first, then by position.
    for i, n in enumerate(argspec.args):
        if n in kwargs:
            pass_args.append(kwargs[n])
            used_kw.add(n)
        elif used_pos < len(args):
            pass_args.append(args[used_pos])
            used_pos += 1
        else:
            unmatched_pos.append(len(pass_args))
            pass_args.append(
                None if i < defaulted_pos else argspec.defaults[i - defaulted_pos]
            )
    # Fill unmatched positional args with unmatched keyword args in order.
    if len(unmatched_pos):
        for k, v in kwargs.items():
            if k in used_kw or k in argspec.kwonlyargs:
                continue
            pass_args[unmatched_pos[0]] = v
            used_kw.add(k)
            unmatched_pos = unmatched_pos[1:]
            if len(unmatched_pos) == 0:
                break
        else:
            if unmatched_pos[0] < defaulted_pos:
                unpassed = ", ".join(
                    argspec.args[u] for u in unmatched_pos if u < defaulted_pos
                )
                raise TypeError(f"{fn.__name__}() cannot be passed {unpassed}.")
    # Pass remaining kw args if they can be accepted.
    pass_kw = {
        k: v
        for k, v in kwargs.items()
        if k not in used_kw and (k in argspec.kwonlyargs or argspec.varargs is not None)
    }
    # Pass remaining positional args if they can be accepted.
    if argspec.varargs is not None:
        pass_args += list(args[used_pos:])
    return fn(*pass_args, **pass_kw)

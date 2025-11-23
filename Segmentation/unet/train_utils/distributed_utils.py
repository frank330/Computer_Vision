"""
分布式训练工具和指标记录模块

该模块提供了用于分布式训练和指标记录的工具类和函数，包括：
- 指标记录和统计（SmoothedValue, MetricLogger）
- 评估指标计算（ConfusionMatrix, DiceCoefficient）
- 分布式训练辅助函数

主要组件：
- SmoothedValue: 平滑值跟踪器，用于记录和统计训练指标
- ConfusionMatrix: 混淆矩阵，用于计算准确率和IoU
- DiceCoefficient: Dice系数计算器
- MetricLogger: 指标记录器，用于记录和打印训练过程
- 分布式训练辅助函数
"""

from collections import defaultdict, deque  # 默认字典和双端队列
import datetime                              # 日期时间处理
import time                                 # 时间相关操作
import torch                                # PyTorch核心库
import torch.nn.functional as F             # PyTorch函数式接口
import torch.distributed as dist            # PyTorch分布式训练模块

import errno                                # 错误号定义
import os                                   # 操作系统接口

from .dice_coefficient_loss import multiclass_dice_coeff, build_target  # Dice系数计算函数


class SmoothedValue(object):
    """
    平滑值跟踪器
    
    该类用于跟踪一系列数值，并提供窗口平滑值或全局平均值。
    常用于记录训练过程中的损失、准确率等指标。
    
    功能：
    - 维护一个固定大小的滑动窗口（deque）
    - 计算窗口内的平均值、中位数、最大值
    - 计算全局平均值
    - 支持多进程同步
    """

    def __init__(self, window_size=20, fmt=None):
        """
        初始化平滑值跟踪器
        
        Args:
            window_size (int): 滑动窗口大小，默认20
            fmt (str, optional): 格式化字符串，用于__str__方法
                默认: "{value:.4f} ({global_avg:.4f})"
        """
        if fmt is None:
            fmt = "{value:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)  # 固定大小的滑动窗口
        self.total = 0.0                          # 所有值的总和
        self.count = 0                            # 值的总个数
        self.fmt = fmt                            # 格式化字符串

    def update(self, value, n=1):
        """
        更新值
        
        Args:
            value (float): 要添加的值
            n (int): 该值出现的次数，默认1
        """
        self.deque.append(value)      # 添加到滑动窗口
        self.count += n               # 更新计数
        self.total += value * n       # 更新总和

    def synchronize_between_processes(self):
        """
        在多进程间同步统计值
        
        注意：该方法只同步count和total，不同步deque（滑动窗口）！
        这是因为deque在多进程间同步比较复杂，且通常只需要全局平均值。
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class ConfusionMatrix(object):
    """
    混淆矩阵类
    
    该类用于计算分割任务的混淆矩阵，并基于混淆矩阵计算各种评估指标：
    - 全局准确率（Global Accuracy）
    - 每个类别的准确率（Per-class Accuracy）
    - 每个类别的IoU（Intersection over Union）
    - 平均IoU（Mean IoU）
    """
    
    def __init__(self, num_classes):
        """
        初始化混淆矩阵
        
        Args:
            num_classes (int): 分割类别数（包括背景）
        """
        self.num_classes = num_classes  # 类别数
        self.mat = None                  # 混淆矩阵，形状为 (num_classes, num_classes)

    def update(self, a, b):
        """
        更新混淆矩阵
        
        Args:
            a (torch.Tensor): 真实标签（Ground Truth），形状为 (N,)
            b (torch.Tensor): 预测标签，形状为 (N,)
                N: 像素总数（通常是 H * W）
        """
        n = self.num_classes
        if self.mat is None:
            # 首次调用时创建混淆矩阵
            self.mat = torch.zeros((n, n), dtype=torch.int64, device=a.device)
        
        with torch.no_grad():
            # 寻找有效像素索引（标签值在有效范围内）
            k = (a >= 0) & (a < n)
            
            # 统计像素真实类别a[k]被预测成类别b[k]的个数
            # 使用巧妙的索引方法：将二维索引 (a, b) 映射为一维索引
            # inds = n * a + b，这样可以唯一标识每个 (真实类别, 预测类别) 对
            inds = n * a[k].to(torch.int64) + b[k]
            
            # 使用bincount统计每个索引出现的次数，然后reshape为混淆矩阵
            self.mat += torch.bincount(inds, minlength=n**2).reshape(n, n)

    def reset(self):
        """
        重置混淆矩阵
        
        将混淆矩阵清零，用于开始新的评估。
        """
        if self.mat is not None:
            self.mat.zero_()

    def compute(self):
        """
        计算评估指标
        
        Returns:
            tuple: (acc_global, acc, iu)
                acc_global: 全局准确率（所有类别）
                acc: 每个类别的准确率，形状为 (num_classes,)
                iu: 每个类别的IoU，形状为 (num_classes,)
        """
        h = self.mat.float()  # 转换为float类型以便计算
        
        # 计算全局预测准确率（混淆矩阵的对角线为预测正确的个数）
        # 全局准确率 = 所有正确预测的像素数 / 总像素数
        acc_global = torch.diag(h).sum() / h.sum()
        
        # 计算每个类别的准确率
        # 类别i的准确率 = 正确预测为类别i的像素数 / 真实标签中类别i的像素数
        acc = torch.diag(h) / h.sum(1)
        
        # 计算每个类别的IoU（Intersection over Union）
        # IoU = 交集 / 并集 = 对角线元素 / (行和 + 列和 - 对角线元素)
        # 行和：真实标签中该类别的像素数
        # 列和：预测结果中该类别的像素数
        iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))
        
        return acc_global, acc, iu

    def reduce_from_all_processes(self):
        """
        在多进程间聚合混淆矩阵
        
        在分布式训练中，将所有进程的混淆矩阵聚合到主进程。
        用于计算全局的评估指标。
        """
        if not torch.distributed.is_available():
            return
        if not torch.distributed.is_initialized():
            return
        torch.distributed.barrier()  # 同步所有进程
        torch.distributed.all_reduce(self.mat)  # 聚合所有进程的混淆矩阵

    def __str__(self):
        acc_global, acc, iu = self.compute()
        return (
            'global correct: {:.1f}\n'
            'average row correct: {}\n'
            'IoU: {}\n'
            'mean IoU: {:.1f}').format(
                acc_global.item() * 100,
                ['{:.1f}'.format(i) for i in (acc * 100).tolist()],
                ['{:.1f}'.format(i) for i in (iu * 100).tolist()],
                iu.mean().item() * 100)


class DiceCoefficient(object):
    """
    Dice系数计算器
    
    该类用于累积计算Dice系数，通常用于验证集评估。
    会忽略背景类别（类别0），只计算前景类别的Dice系数。
    """
    
    def __init__(self, num_classes: int = 2, ignore_index: int = -100):
        """
        初始化Dice系数计算器
        
        Args:
            num_classes (int): 分割类别数（包括背景），默认2
            ignore_index (int): 需要忽略的像素值，默认-100
        """
        self.cumulative_dice = None      # 累积的Dice系数总和
        self.num_classes = num_classes    # 类别数
        self.ignore_index = ignore_index  # 忽略索引
        self.count = None                 # 更新次数

    def update(self, pred, target):
        """
        更新Dice系数
        
        Args:
            pred (torch.Tensor): 模型预测结果（logits），形状为 (N, C, H, W)
            target (torch.Tensor): 真实标签，形状为 (N, H, W)
        """
        if self.cumulative_dice is None:
            self.cumulative_dice = torch.zeros(1, dtype=pred.dtype, device=pred.device)
        if self.count is None:
            self.count = torch.zeros(1, dtype=pred.dtype, device=pred.device)
        
        # 计算Dice系数，忽略背景类别（类别0）
        # 将预测结果转换为one-hot编码
        pred = F.one_hot(pred.argmax(dim=1), self.num_classes).permute(0, 3, 1, 2).float()
        # 构建目标标签（one-hot编码）
        dice_target = build_target(target, self.num_classes, self.ignore_index)
        # 计算前景类别的Dice系数（忽略背景类别0）
        self.cumulative_dice += multiclass_dice_coeff(pred[:, 1:], dice_target[:, 1:], ignore_index=self.ignore_index)
        self.count += 1

    @property
    def value(self):
        """
        获取平均Dice系数
        
        Returns:
            torch.Tensor: 平均Dice系数（累积Dice系数 / 更新次数）
        """
        if self.count == 0:
            return 0
        else:
            return self.cumulative_dice / self.count

    def reset(self):
        """
        重置Dice系数计算器
        
        将累积的Dice系数和计数清零，用于开始新的评估。
        """
        if self.cumulative_dice is not None:
            self.cumulative_dice.zero_()

        if self.count is not None:
            self.count.zeros_()

    def reduce_from_all_processes(self):
        """
        在多进程间聚合Dice系数
        
        在分布式训练中，将所有进程的Dice系数聚合到主进程。
        用于计算全局的Dice系数。
        """
        if not torch.distributed.is_available():
            return
        if not torch.distributed.is_initialized():
            return
        torch.distributed.barrier()  # 同步所有进程
        torch.distributed.all_reduce(self.cumulative_dice)  # 聚合累积Dice系数
        torch.distributed.all_reduce(self.count)  # 聚合计数


class MetricLogger(object):
    """
    指标记录器
    
    该类用于记录和打印训练过程中的各种指标（损失、准确率、学习率等）。
    支持自动格式化输出，包括进度条、ETA（预计剩余时间）等信息。
    """
    
    def __init__(self, delimiter="\t"):
        """
        初始化指标记录器
        
        Args:
            delimiter (str): 打印指标时的分隔符，默认"\t"（制表符）
        """
        self.meters = defaultdict(SmoothedValue)  # 指标字典，每个指标对应一个SmoothedValue
        self.delimiter = delimiter                  # 分隔符

    def update(self, **kwargs):
        """
        更新指标值
        
        Args:
            **kwargs: 要更新的指标，例如 loss=0.5, lr=0.01
        """
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()  # 将Tensor转换为Python数值
            assert isinstance(v, (float, int))
            self.meters[k].update(v)  # 更新对应指标

    def __getattr__(self, attr):
        """
        获取指标对象（属性访问）
        
        允许通过属性访问方式获取指标，例如：logger.loss
        
        Args:
            attr (str): 指标名称
        
        Returns:
            SmoothedValue: 对应的指标对象
        """
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        """
        格式化输出所有指标
        
        Returns:
            str: 格式化的指标字符串
        """
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        """
        在多进程间同步所有指标
        
        在分布式训练中，将所有进程的指标同步到主进程。
        """
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        """
        添加新的指标跟踪器
        
        Args:
            name (str): 指标名称
            meter (SmoothedValue): 指标跟踪器对象
        """
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        """
        迭代数据加载器并定期打印训练信息
        
        该函数是一个生成器，用于迭代数据加载器，并在指定频率打印训练信息。
        打印的信息包括：进度、ETA、指标值、时间、显存使用等。
        
        Args:
            iterable: 可迭代对象（通常是DataLoader）
            print_freq (int): 打印频率（每N个batch打印一次）
            header (str, optional): 打印信息的标题，默认None
        
        Yields:
            数据项：从iterable中yield的数据项
        """
        i = 0
        if not header:
            header = ''
        start_time = time.time()  # 记录开始时间
        end = time.time()         # 记录上次迭代结束时间
        iter_time = SmoothedValue(fmt='{avg:.4f}')  # 迭代时间跟踪器
        data_time = SmoothedValue(fmt='{avg:.4f}')  # 数据加载时间跟踪器
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'  # 格式化进度数字的格式
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {}'.format(header, total_time_str))


def mkdir(path):
    """
    创建目录（如果不存在）
    
    该函数创建指定的目录，如果目录已存在则不会报错。
    
    Args:
        path (str): 要创建的目录路径
    """
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:  # 如果错误不是"目录已存在"，则抛出异常
            raise


def setup_for_distributed(is_master):
    """
    设置分布式训练环境
    
    该函数在非主进程（master process）中禁用print输出，避免多进程训练时输出混乱。
    只有主进程（rank 0）会打印信息。
    
    Args:
        is_master (bool): 是否为主进程（rank == 0）
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    """
    检查分布式训练是否可用且已初始化
    
    Returns:
        bool: True表示分布式训练可用且已初始化，False表示不可用或未初始化
    """
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    """
    获取分布式训练的世界大小（总进程数）
    
    Returns:
        int: 总进程数，如果不是分布式训练则返回1
    """
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    """
    获取当前进程的rank（进程编号）
    
    Returns:
        int: 当前进程的rank，主进程为0，如果不是分布式训练则返回0
    """
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    """
    检查当前进程是否为主进程
    
    Returns:
        bool: True表示是主进程（rank == 0），False表示不是
    """
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    """
    只在主进程上保存文件
    
    该函数确保在多进程训练时，只有主进程会保存模型等文件，避免重复保存。
    
    Args:
        *args, **kwargs: 传递给torch.save的参数
    """
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    """
    初始化分布式训练模式
    
    该函数根据环境变量或参数初始化分布式训练。
    支持多种分布式训练启动方式：
    1. PyTorch分布式启动（torchrun）
    2. SLURM集群调度系统
    3. 手动指定rank
    
    Args:
        args: 参数对象，需要包含以下属性：
            - dist_url: 分布式训练初始化URL
            - rank (可选): 进程rank
            - world_size (可选): 总进程数
            - gpu (可选): GPU设备编号
    
    环境变量：
        - RANK: 进程rank（PyTorch分布式）
        - WORLD_SIZE: 总进程数（PyTorch分布式）
        - LOCAL_RANK: 本地GPU编号（PyTorch分布式）
        - SLURM_PROCID: SLURM进程ID
    """
    # 方式1：PyTorch分布式启动（torchrun）
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    # 方式2：SLURM集群调度系统
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    # 方式3：手动指定rank（参数中已有）
    elif hasattr(args, "rank"):
        pass
    # 方式4：不使用分布式训练
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    # 启用分布式训练
    args.distributed = True

    # 设置当前进程使用的GPU设备
    torch.cuda.set_device(args.gpu)
    # 设置分布式后端（NCCL用于多GPU训练）
    args.dist_backend = 'nccl'
    
    # 打印分布式初始化信息
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    
    # 初始化进程组
    torch.distributed.init_process_group(
        backend=args.dist_backend,      # 后端：NCCL
        init_method=args.dist_url,      # 初始化方法（URL）
        world_size=args.world_size,     # 总进程数
        rank=args.rank                  # 当前进程rank
    )
    
    # 设置只在主进程打印输出
    setup_for_distributed(args.rank == 0)

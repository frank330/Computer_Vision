"""
解析训练结果文件并绘制折线图
"""
import re
import matplotlib.pyplot as plt
import numpy as np

def parse_results_file(file_path):
    """
    解析训练结果文件，提取每个epoch的指标
    
    Args:
        file_path: 结果文件路径
        
    Returns:
        dict: 包含各个指标的字典
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 提取epoch、train_loss、lr、dice coefficient、mean IoU
    epochs = []
    train_losses = []
    lrs = []
    dice_coefficients = []
    mean_ious = []
    
    # 使用正则表达式匹配每个epoch的数据
    pattern = r'\[epoch: (\d+)\]\ntrain_loss: ([\d.]+)\nlr: ([\d.]+)\ndice coefficient: ([\d.]+).*?mean IoU: ([\d.]+)'
    matches = re.findall(pattern, content, re.DOTALL)
    
    for match in matches:
        epoch, train_loss, lr, dice, mean_iou = match
        epochs.append(int(epoch))
        train_losses.append(float(train_loss))
        lrs.append(float(lr))
        dice_coefficients.append(float(dice))
        mean_ious.append(float(mean_iou))
    
    return {
        'epochs': epochs,
        'train_losses': train_losses,
        'lrs': lrs,
        'dice_coefficients': dice_coefficients,
        'mean_ious': mean_ious
    }


def plot_training_curves(data, save_path='training_curves.png'):
    """
    绘制训练曲线图
    
    Args:
        data: 解析后的数据字典
        save_path: 保存图片的路径
    """
    epochs = data['epochs']
    train_losses = data['train_losses']
    mean_ious = data['mean_ious']
    
    # 创建图表，使用中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']  # 支持中文
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # 绘制train_loss折线图
    ax1.plot(epochs, train_losses, 'b-o', linewidth=2, markersize=6, label='Train Loss')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Train Loss', fontsize=12)
    ax1.set_title('训练损失曲线 (Train Loss)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    ax1.set_xlim(left=-0.5, right=max(epochs) + 0.5)
    
    # 在图上标注数值
    for i, (epoch, loss) in enumerate(zip(epochs, train_losses)):
        if i % max(1, len(epochs) // 10) == 0 or i == len(epochs) - 1:  # 只标注部分点，避免过于拥挤
            ax1.annotate(f'{loss:.4f}', (epoch, loss), 
                        textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
    
    # 绘制mean IoU折线图（mean_loss可能指的是这个）
    ax2.plot(epochs, mean_ious, 'r-s', linewidth=2, markersize=6, label='Mean IoU')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Mean IoU (%)', fontsize=12)
    ax2.set_title('平均IoU曲线 (Mean IoU)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    ax2.set_xlim(left=-0.5, right=max(epochs) + 0.5)
    
    # 在图上标注数值
    for i, (epoch, iou) in enumerate(zip(epochs, mean_ious)):
        if i % max(1, len(epochs) // 10) == 0 or i == len(epochs) - 1:
            ax2.annotate(f'{iou:.1f}%', (epoch, iou), 
                        textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"图表已保存到: {save_path}")
    plt.show()


def plot_combined_curves(data, save_path='training_combined.png'):
    """
    在同一张图上绘制train_loss和mean IoU（使用双y轴）
    
    Args:
        data: 解析后的数据字典
        save_path: 保存图片的路径
    """
    epochs = data['epochs']
    train_losses = data['train_losses']
    mean_ious = data['mean_ious']
    
    # 创建图表，使用中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # 左y轴：train_loss
    color1 = 'tab:blue'
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Train Loss', color=color1, fontsize=12)
    line1 = ax1.plot(epochs, train_losses, 'b-o', linewidth=2, markersize=6, label='Train Loss', color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(left=-0.5, right=max(epochs) + 0.5)
    
    # 右y轴：mean IoU
    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel('Mean IoU (%)', color=color2, fontsize=12)
    line2 = ax2.plot(epochs, mean_ious, 'r-s', linewidth=2, markersize=6, label='Mean IoU', color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # 添加标题和图例
    plt.title('训练损失和平均IoU曲线', fontsize=14, fontweight='bold', pad=20)
    
    # 合并图例
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"组合图表已保存到: {save_path}")
    plt.show()


if __name__ == '__main__':
    # 解析结果文件
    results_file = 'results20251119-090807.txt'
    print(f"正在解析文件: {results_file}")
    
    try:
        data = parse_results_file(results_file)
        
        print(f"\n成功解析 {len(data['epochs'])} 个epoch的数据")
        print(f"Epoch范围: {min(data['epochs'])} - {max(data['epochs'])}")
        print(f"Train Loss范围: {min(data['train_losses']):.4f} - {max(data['train_losses']):.4f}")
        print(f"Mean IoU范围: {min(data['mean_ious']):.1f}% - {max(data['mean_ious']):.1f}%")
        
        # 绘制分离的图表
        print("\n正在绘制分离的训练曲线...")
        plot_training_curves(data, 'training_curves.png')
        
        # 绘制组合图表
        print("\n正在绘制组合的训练曲线...")
        plot_combined_curves(data, 'training_combined.png')
        
    except FileNotFoundError:
        print(f"错误: 找不到文件 {results_file}")
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()


# import random
# import matplotlib.pyplot as plt
# import numpy as np

# # -------------------------- 1. 全局参数配置 --------------------------
# # 核心参数（统一配置，便于调整）
# T0 = 0.413          # 初始Ti值
# TPR = 0.25          # 目标PR值
# num_steps = 100     # 训练步数
# PRi_range = (0.0, 0.5)  # PRi随机生成范围
# random_seeds = [42, 123, 456]  # 3个不同随机种子（对应3种PRi初始化）
# base = 0.9   # 衰减系数（用于PRi生成）
# threshold = 0.05    # 衰减阈值

# # 颜色配置（学术配色，区分度高）
# colors = ['#2E86AB', '#A23B72', '#F18F01']  # 蓝、紫、橙
# labels = ['<f_h>','<butin>','<dunpai>']  # 每条线的标签

# # 存储所有结果（3组：PRi_list, Ti_list, τi_list）
# all_PRi = []
# all_Ti = []

# # -------------------------- 2. 多组数据生成与核心计算 --------------------------
# for seed in random_seeds:
#     # 固定当前种子，生成独立的PRi序列（3种初始化）
#     random.seed(seed)
#     # 计算衰减系数ratio
#     aug=random.uniform(-0.05,0.05)
#     ratio_decay=base+aug
#     print(ratio_decay)
#     ratio = [ratio_decay**i if (ratio_decay**i) > threshold+random.uniform(-0.025,0.025) else threshold+random.uniform(-0.025,0.025) for i in range(num_steps)]
#     # 生成当前组的PRi（按原公式，保证每组独立）
#     PRi_list = [TPR + ratio[i] * (random.uniform(*PRi_range) - 0.25) for i in range(num_steps)]
    
#     # 初始化当前组的Ti和τi
#     Ti_list = [T0]
#     τi_list = [T0]
    
#     # 核心计算逻辑（与原逻辑一致）
#     for i in range(1, num_steps):
#         PRi = PRi_list[i]
#         Δi = TPR - PRi
#         Ti_prev = Ti_list[i-1]
#         τi_prev = τi_list[i-1]
        
#         # 计算Σ（符号判断）
#         Σ = 2 if (τi_prev * Δi) > 0 else 0
        
#         # 计算当前步Ti和τi
#         Ti = Ti_prev + 0.12 * Δi - Σ * τi_prev * Δi
#         τi = 0.8 * τi_prev + 0.2 * (Ti - Ti_prev)
        
#         Ti_list.append(Ti)
#         τi_list.append(τi)
    
#     # 存储当前组结果
#     all_PRi.append(PRi_list)
#     all_Ti.append(Ti_list)

# steps = np.arange(num_steps)  # 步长序号（0~99）

# # -------------------------- 独立图片1：Ti变化趋势 --------------------------

# fig1, ax1 = plt.subplots(figsize=(10, 3.5), dpi=100)

# for idx, (Ti_list, color, label) in enumerate(zip(all_Ti, colors, labels)):
#     ax1.plot(steps, Ti_list, markersize=3, linewidth=2,
#              color=color, label=label, markerfacecolor='white', markeredgewidth=1)

# # 子图1美化
# ax1.set_xlabel('Training Step', fontsize=12, fontweight='medium')
# ax1.set_ylabel('$T_i$', fontsize=12, fontweight='medium')
# ax1.set_title('$T_i$ Variation Trend', fontsize=14, fontweight='bold', pad=15)
# ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
# ax1.legend(loc='upper right', frameon=True, fancybox=True, shadow=False, fontsize=10)
# ax1.tick_params(axis='both', which='major', labelsize=10)

# # 优化x轴刻度
# ax1.set_xticks(steps[::5])
# ax1.set_xlim(0, num_steps-1)

# plt.tight_layout()
# # 保存Ti独立图片
# plt.savefig('Ti_variation_trend.png', dpi=300, bbox_inches='tight')
# plt.show()  # 显示图片（可注释掉不显示，仅保存）

# # -------------------------- 独立图片2：PRi变化趋势 --------------------------
# fig2, ax2 = plt.subplots(figsize=(10, 3.5), dpi=100)

# for idx, (PRi_list, color, label) in enumerate(zip(all_PRi, colors, labels)):
#     ax2.plot(steps, PRi_list, markersize=3, linewidth=2,
#              color=color, label=label, markerfacecolor='white', markeredgewidth=1)

# # 标注TPR目标线（参考基准）
# ax2.axhline(y=TPR, color='#7209B7', linestyle='--', linewidth=1.5,
#             label=f'Target TPR = {TPR}', alpha=0.8)

# # 子图2美化
# ax2.set_xlabel('Training Step', fontsize=12, fontweight='medium')
# ax2.set_ylabel('$PR_i$', fontsize=12, fontweight='medium')
# ax2.set_title('$PR_i$ Variation Trend', fontsize=14, fontweight='bold', pad=15)
# ax2.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
# ax2.legend(loc='upper right', frameon=True, fancybox=True, shadow=False, fontsize=10)
# ax2.tick_params(axis='both', which='major', labelsize=10)

# # 优化x轴刻度
# ax2.set_xticks(steps[::5])
# ax2.set_xlim(0, num_steps-1)

# plt.tight_layout()
# # 保存PRi独立图片
# plt.savefig('PRi_variation_trend.png', dpi=300, bbox_inches='tight')
# plt.show()  # 显示图片（可注释掉不显示，仅保存）

# # -------------------------- 4. 输出关键统计结果（可选） --------------------------
# print("="*80)
# print("Key Statistics (Final Step Values)")
# print("="*80)
# print(f"{'Initialization':<15} {'Final Ti':<12} {'Final PRi':<12} {'Ti Std (All Steps)':<15}")
# print("-"*80)
# for label, Ti_list, PRi_list in zip(labels, all_Ti, all_PRi):
#     final_Ti = Ti_list[-1]
#     final_PRi = PRi_list[-1]
#     Ti_std = np.std(Ti_list)
#     print(f"{label:<15} {final_Ti:.6f} {'':<2} {final_PRi:.6f} {'':<2} {Ti_std:.6f}")
# print("="*80)



import numpy as np
import matplotlib.pyplot as plt
import random

# -------------------------- 1. 参数配置 --------------------------
num_steps = 100  # 训练步数
labels = ['<f_h>_Fast Convergence', '<f_h>_Slow Convergence']  # 两条曲线标签（基于要求扩展）
init_range = (800, 1200)  # 初始Loss随机范围
final_range = (-50, 50)  # 最终收敛震荡范围
random_seed = 42  # 随机种子（保证可复现）

# 颜色配置（学术配色，区分度高）
colors = ['#2E86AB', '#A23B72']
# 异常值配置（模拟训练不稳定）
outlier_steps = [13, 31, 63, 84,90]  # 异常值出现的步长
outlier_magnitude = (-200, 200)  # 异常值幅度

# -------------------------- 2. 生成Loss曲线数据 --------------------------
np.random.seed(random_seed)
random.seed(random_seed)

# 初始化两条曲线的初始Loss（800-1200随机实数）
loss_fast = [np.random.uniform(*init_range)]
loss_slow = [np.random.uniform(*init_range)]

# 生成每条曲线的衰减系数（控制收敛速度）
for step in range(1, num_steps):
    # -------------------------- 快速收敛曲线 --------------------------
    # 核心衰减逻辑：前期快速下降，后期小幅度震荡
    decay_rate_fast = 0.85 if step < 30 else 0.98  # 前30步快速衰减，之后缓慢衰减
    noise_fast = np.random.normal(0, 30)  # 基础噪声（模拟不稳定性）
    
    # 计算当前步Loss（基于前一步衰减）
    current_loss_fast = loss_fast[-1] * decay_rate_fast + noise_fast
    
    # 加入异常值（随机触发，模拟训练波动）
    if step in outlier_steps and random.random() > 0.7:
        current_loss_fast += np.random.uniform(*outlier_magnitude)
    
    # 收敛后限制在[-50,50]震荡
    if abs(current_loss_fast) < 50:
        current_loss_fast = np.random.uniform(*final_range)
    
    # -------------------------- 慢速收敛曲线 --------------------------
    # 核心衰减逻辑：全程缓慢下降，后期才收敛
    decay_rate_slow = 0.95 if step < 60 else 0.99  # 前60步缓慢衰减，之后微调
    noise_slow = np.random.normal(0, 45)  # 更大的噪声（收敛更慢，波动更大）
    
    # 计算当前步Loss（基于前一步衰减）
    current_loss_slow = loss_slow[-1] * decay_rate_slow + noise_slow
    
    # 加入异常值（触发概率更高，波动更明显）
    if step in outlier_steps and random.random() > 0.5:
        current_loss_slow += np.random.uniform(*outlier_magnitude)
    
    # 收敛后限制在[-50,50]震荡
    if abs(current_loss_slow) < 50:
        current_loss_slow = np.random.uniform(*final_range)
    
    # 存入列表
    loss_fast.append(current_loss_fast)
    loss_slow.append(current_loss_slow)

# 整合所有Loss数据
all_loss = [loss_fast, loss_slow]
steps = np.arange(num_steps)



# 创建画布（学术图表尺寸）
fig, ax = plt.subplots(figsize=(10, 4), dpi=100)

# 绘制两条Loss曲线
for idx, (loss_data, color, label) in enumerate(zip(all_loss, colors, labels)):
    ax.plot(steps, loss_data, linewidth=2.5, color=color, label=label,
            marker='o', markersize=3, markerfacecolor='white', markeredgewidth=1.5)

# 标注收敛目标区间（[-50,50]）
# ax.axhspan(ymin=-50, ymax=50, color='#F1F1F1', alpha=0.5, label='Convergence Range [-50, 50]')
ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.3)  # 零基准线

# 图表美化（学术规范）
ax.set_xlabel('Training Step', fontsize=12, fontweight='medium')
ax.set_ylabel('Loss Value', fontsize=12, fontweight='medium')
ax.set_title('Loss Variation Trend (Fast vs. Slow Convergence)', fontsize=14, fontweight='bold', pad=15)

# 坐标轴优化
ax.set_xticks(steps[::5])  # 每5步显示一个刻度
ax.set_xlim(0, num_steps-1)
ax.tick_params(axis='both', which='major', labelsize=10)

# 网格与图例
ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=False, fontsize=10)

# 调整布局（避免标签截断）
plt.tight_layout()

# 保存图片（论文常用格式）
plt.savefig('loss_convergence_trend.png', dpi=300, bbox_inches='tight')

# 显示图片
plt.show()

# -------------------------- 4. 输出关键统计信息 --------------------------
print("="*80)
print("Loss Curve Statistics")
print("="*80)
print(f"{'Curve Type':<25} {'Initial Loss':<15} {'Final Loss':<15} {'Converge Step':<15}")
print("-"*80)

# 计算收敛步数（首次进入[-50,50]并稳定的步长）
def get_converge_step(loss_list):
    for step, loss in enumerate(loss_list):
        if abs(loss) <= 50:
            # 验证后续5步是否稳定在区间内（避免偶然波动）
            if step + 5 < num_steps and all(abs(l) <= 50 for l in loss_list[step:step+5]):
                return step
    return "Not Converged"  # 理论上不会触发

for label, loss_data in zip(labels, all_loss):
    init_loss = loss_data[0]
    final_loss = loss_data[-1]
    converge_step = get_converge_step(loss_data)
    print(f"{label:<25} {init_loss:.2f} {'':<3} {final_loss:.2f} {'':<3} {converge_step}")
print("="*80)
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ⚠️ 在一切开始前，锁死随机性！
set_seed(3407)
# 1. 环境还是你那个环境（原封不动）
rewards = np.array([
    [  0,   0,   0],
    [  0, -50, -50],
    [  0,   0, 100],
])
gamma = 0.8
actions = [[-1,0],[1,0],[0,-1],[0,1]]

# ==========================================
# 核心大换血 1：用黑盒取代 value=np.zeros()
# ==========================================
class VNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 16),  # 眼睛：输入2个数字（坐标 r 和 c）
            nn.ReLU(),         # 联想开关
            nn.Linear(16, 16), # 大脑皮层
            nn.ReLU(),
            nn.Linear(16, 1)   # 嘴巴：输出1个数字（身价 Value）
        )
        
    def forward(self, r, c):
        # 把坐标打包成张量 (Tensor) 喂给黑盒
        state = torch.tensor([float(r), float(c)])
        return self.net(state)

# 实例化大脑、设置会计（优化器）和判卷标准（Loss函数）
model = VNetwork()
optimizer = optim.Adam(model.parameters(), lr=0.001) # 学习率 0.01
loss_fn = nn.MSELoss() # 均方误差

# ==========================================
# 核心大换血 2：开始训练 (以前是 for i in range(200))
# ==========================================
for i in range(2001): # 神经网络比较笨，得多教几百圈
    total_loss = 0
    
    # 依然是遍历全图扫每一个格子
    for r in range(rewards.shape[0]):
        for c in range(rewards.shape[1]):
            
            # 【第 1 步：算出一笔“铁证如山”的目标账 (Target)】
            if rewards[r][c] == -50 or rewards[r][c] == 100:
                target_v = torch.tensor([float(rewards[r][c])])
            else:
                # 查四周，找最大的未来价值
                possible_next_values = []
                for action in actions:
                    next_r, next_c = r + action[0], c + action[1]
                    # 撞墙处理：还在原地
                    if next_r < 0 or next_r >= 3 or next_c < 0 or next_c >= 3:
                        next_r, next_c = r, c
                    
                    # ⚠️ 极其关键：这里不再是查老账本 value[next_r][next_c] 了！
                    # 而是去问神经网络：“嘿，你觉得旁边的格子值多少钱？”
                    with torch.no_grad(): # （这里只是问问题，不准算误差求导）
                        next_v = model(next_r, next_c).item()
                    possible_next_values.append(next_v)
                
                # 算出绝对真理 (Target) = 现在的钱 + 0.8 * 未来最赚钱的一步
                target_v = torch.tensor([rewards[r][c] + gamma * max(possible_next_values)])
            
            # 【第 2 步：神经网络神圣 5 步曲！】
            optimizer.zero_grad()              # 1. 撕掉旧账本 (清空梯度)
            pred_v = model(r, c)               # 2. 黑盒对当前的 (r, c) 进行盲猜
            loss = loss_fn(pred_v, target_v)   # 3. 判卷算分 (比较盲猜和真理的差距)
            loss.backward()                    # 4. 自动找茬求导 (反向传播)
            optimizer.step()                   # 5. 微调大脑里的参数！
            
            total_loss += loss.item()
            
    # 每 100 圈打印一次成绩报告
    if i % 100 == 0:
        print(f"第 {i} 圈，总误差(Loss): {total_loss:.4f}")

# ==========================================
# 训练结束！我们来考考它最终学成了什么样？
# ==========================================
print("\n---使用深度算法---")
for r in range(3):
    row_values = []
    for c in range(3):
        # 此时不需要查表了，直接把坐标输入给大脑，让它当场算！
        val = model(r, c).item()
        row_values.append(f"{val:6.1f}")
    print("[" + ", ".join(row_values) + "]")
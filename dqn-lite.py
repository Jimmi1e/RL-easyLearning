import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import collections # 引入收纳箱
import matplotlib.pyplot as plt
from IPython import display

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
set_seed(42)

# ==========================================
# 🌍 环境
# ==========================================
class MazeEnv:
    def __init__(self, reward_map):
        self.rewards = np.array(reward_map)
        self.rows, self.cols = self.rewards.shape # 自动获取尺寸
        self.actions = [[-1,0], [1,0], [0,-1], [0,1]] 
        self.r, self.c = 0, 0

    def reset(self):
        self.r, self.c = 0, 0
        return self.r, self.c

    def step(self, action_idx):
        move = self.actions[action_idx]
        next_r, next_c = self.r + move[0], self.c + move[1]
        
        # 只要在这里使用 self.rows 和 self.cols，无论地图多大都完美适配！
        if 0 <= next_r < self.rows and 0 <= next_c < self.cols:
            self.r, self.c = next_r, next_c
            
        reward = self.rewards[self.r][self.c]
        done = (reward == 100 or reward == -50)
        return self.r, self.c, reward, done



# ==========================================
# 经验回放池 (错题本)
# ==========================================
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = collections.deque(maxlen=capacity) 

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done)) 

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size) 
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    def __len__(self):
        return len(self.buffer)

# ==========================================
# 🧠 支持 Batch 处理的DQN
# ==========================================
class QNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 4)
        )
    def forward(self, state):
        # 直接接收打包好的 state (单步或批量都可以)
        return self.net(state)
    

# ==========================================
# 📊 画图
# ==========================================
class TrainingMonitor:
    def __init__(self):
        self.rewards = []
        self.losses = []

    def record(self, reward, loss):
        self.rewards.append(reward)
        self.losses.append(loss)

    def plot_results(self):
        # 训练结束后一次性调用，生成高质量图表
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.rewards, color='blue', alpha=0.7)
        plt.title("Episodic Rewards")
        plt.xlabel("Episode")
        
        plt.subplot(1, 2, 2)
        plt.plot(self.losses, color='red', alpha=0.7)
        plt.title("Loss (Moving Average)")
        plt.xlabel("Episode")
        
        plt.tight_layout()
        plt.show() # 在 VS Code 中这会弹出一个稳定的窗口



# ==========================================
# ⚔️ 准备
# ==========================================
my_map = [
    [0, 0, 0, 0, 0],
    [0, -50, 0, -50, 0],
    [0, 0, 0, 0, 0],
    [0, -50, 0, -50, 0],
    [0, 0, 0, 0, 100]
]
env = MazeEnv(my_map)
model = QNetwork()
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()
gamma = 0.85
buffer = ReplayBuffer(capacity=5000) 
batch_size = 32 
best_reward = -float('inf') # 初始值设为无穷大，保证第一次一定能被替换
save_path = "best_model.pth"
# 初始化监控器
monitor = TrainingMonitor()
# ==========================================
# DQN 核心循环：分离探索与学习
# ==========================================
for episode in range(1000):
    r, c = env.reset()
    done = False
    total_reward = 0  
    total_loss = 0    
    loss_count = 0
    while not done:
        # --- 1. 徒弟做决定 ---
        state = torch.tensor([float(r)/env.rows, float(c)/env.cols], dtype=torch.float32)
        q_values = model(state)
        
        if random.random() < 0.4:
            action = random.randint(0, 3)
        else:
            action = torch.argmax(q_values).item()
            
        # --- 2. 现实铁拳 ---
        next_r, next_c, reward, done = env.step(action)
        total_reward += reward
        # --- 3. 记入错题本 (绝对不马上学习！) ---
        buffer.push([r/env.rows, c/env.cols], action, reward, [next_r/env.rows, next_c/env.cols], done)
        
        # --- 4. 攒够复习抽查 ---
        if len(buffer) >= batch_size:
            b_state, b_action, b_reward, b_next_state, b_done = buffer.sample(batch_size)
            
            b_state = torch.tensor(b_state, dtype=torch.float32)
            b_next_state = torch.tensor(b_next_state, dtype=torch.float32)
            b_action = torch.tensor(b_action, dtype=torch.int64).unsqueeze(1) 
            b_reward = torch.tensor(b_reward, dtype=torch.float32)
            b_done = torch.tensor(b_done, dtype=torch.float32)

            # 徒弟预测当前状态的所有 Q 值
            q_evals = model(b_state)
            # gather 魔法：拔出实际执行的那个动作的 Q 值
            q_eval = q_evals.gather(1, b_action).squeeze(1) 
            
            with torch.no_grad():
                # 师傅预测下一步的最高 Q 值
                q_nexts = model(b_next_state)
                q_next_max = q_nexts.max(1)[0] 
                
            # 死亡掩码公式：死了就没未来了
            q_target = b_reward + gamma * q_next_max * (1 - b_done)
            
            # 神圣微积分
            loss = loss_fn(q_eval, q_target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            loss_count += 1

        r, c = next_r, next_c

    avg_loss = total_loss / loss_count if loss_count > 0 else 0
    monitor.record(total_reward, avg_loss)
    if total_reward >= best_reward:
        best_reward = total_reward
        torch.save(model.state_dict(), save_path)
        print(f"🏆 新纪录！模型已保存，当前最高得分: {best_reward}")
print("✅ DQN 训练彻底完成！")
monitor.plot_results()

# ==========================================
# 🎓 终极验收考试：纯凭实力走迷宫！
# ==========================================
# print("\n--- 🚀 考核开始：AI 亲自下场 ---")

# r, c = env.reset()
# done = False
# step_count = 0
# path = [(r, c)] 

# while not done and step_count < 40: 
    
#     with torch.no_grad(): 
#         # ⚠️ 修复点：包装成 Tensor 传给大脑
#         state = torch.tensor([float(r)/env.rows, float(c)/env.cols], dtype=torch.float32)
#         q_values = model(state)
        
#     action = torch.argmax(q_values).item()
#     next_r, next_c, reward, done = env.step(action)
#     path.append((next_r, next_c))
    
#     r, c = next_r, next_c
#     step_count += 1

# path_str = " -> ".join([f"({pr},{pc})" for pr, pc in path])
# print(f"📍 AI 走出的路线: {path_str}")

# if reward == 100:
#     print(f"🏆 评价：绝世高手！仅用 {step_count} 步完美避开陷阱，拿到 100 分通关！")
# elif reward == -50:
#     print(f"💀 评价：人工智障！一头扎进了陷阱，卒。")
# else:
    # print(f"🌀 评价：迷失鬼打墙！绕了 {step_count} 步都没走到终点，饿死了。")
# ==========================================
# 🎓 终极验收考试：纯凭实力走迷宫！
# ==========================================
print("\n--- 🚀 考核开始：加载最强巅峰大脑，AI 亲自下场 ---")

# 🌟 新增逻辑：读取保存的巅峰状态权重
best_model = QNetwork() # 重新实例化一个干净的大脑
try:
    best_model.load_state_dict(torch.load(save_path))
    print(f"✅ 成功加载最优权重文件: {save_path}")
except FileNotFoundError:
    print(f"⚠️ 未找到 {save_path}，将使用当前最终模型进行测试")
    best_model = model 

# 切换到测试模式 (虽然你现在没用 Dropout/BatchNorm，但这是工业界必须养成的绝对好习惯)
best_model.eval() 

r, c = env.reset()
done = False
step_count = 0
path = [(r, c)] 

while not done and step_count < 40: 
    
    with torch.no_grad(): 
        # 包装成 Tensor 传给大脑
        state = torch.tensor([float(r)/env.rows, float(c)/env.cols], dtype=torch.float32)
        # ⚠️ 注意这里：使用的是 best_model，而不是训练循环里的 model
        q_values = best_model(state)
        
    action = torch.argmax(q_values).item()
    next_r, next_c, reward, done = env.step(action)
    path.append((next_r, next_c))
    
    r, c = next_r, next_c
    step_count += 1

path_str = " -> ".join([f"({pr},{pc})" for pr, pc in path])
print(f"📍 AI 走出的路线: {path_str}")

if reward == 100:
    print(f"🏆 评价：绝世高手！仅用 {step_count} 步完美避开陷阱，拿到 100 分通关！")
elif reward == -50:
    print(f"💀 评价：人工智障！一头扎进了陷阱，卒。")
else:
    print(f"🌀 评价：迷失鬼打墙！绕了 {step_count} 步都没走到终点，饿死了。")


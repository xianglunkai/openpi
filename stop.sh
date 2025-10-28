# # Release GPU memory
# python -c "import torch; [torch.cuda.empty_cache() for _ in range(torch.cuda.device_count())]"

python -c "import torch; [torch.cuda.empty_cache() for _ in [0,1,2,3,4,5,6,7]]" 
python -c "import torch; torch.cuda.empty_cache()" &> /dev/null


# 精确匹配 conda 环境路径
kill $(ps aux | grep '/workspace/miniconda/envs/lerobot/bin/python' | awk '{print $2}') 
kill $(ps aux | grep '/workspace/miniconda/envs/lerobot/bin/torchrun' | awk '{print $2}')


# # 直接通过进程名匹配并杀死所有 python3.10 进程（root 用户无需 sudo）
pkill -9 -f python

# # 若仍有残留，通过 PID 强制杀死（针对 fuser 列出的进程）
# for gpu in $(seq 0 7); do
#   fuser -v /dev/nvidia$gpu 2>/dev/null | grep -v grep | awk '{for(i=1;i<=NF;i++) print $i}' | xargs -r kill -9 2>/dev/null
# done
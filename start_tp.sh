### 杀掉之前的xllm服务化进程
pkill -9 xllm

# 0. 加载 Ascend 环境（必须先于 python3 调用）
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver:$LD_LIBRARY_PATH

# 1. 环境变量设置
export PYTHON_INCLUDE_PATH="$(python3 -c 'from sysconfig import get_paths; print(get_paths()["include"])')"
export PYTHON_LIB_PATH="$(python3 -c 'from sysconfig import get_paths; print(get_paths()["include"])')"
export PYTORCH_NPU_INSTALL_PATH=/usr/local/libtorch_npu/  # NPU 版 PyTorch 路径
export PYTORCH_INSTALL_PATH="$(python3 -c 'import torch, os; print(os.path.dirname(os.path.abspath(torch.__file__)))')"  # PyTorch 安装路径
export LIBTORCH_ROOT="$PYTORCH_INSTALL_PATH"  # LibTorch 路径
export LD_LIBRARY_PATH=/usr/local/libtorch_npu/lib:$LD_LIBRARY_PATH  # 添加 NPU 库路径

# 2. NPU-device
# export ASCEND_RT_VISIBLE_DEVICES=4,5

export ASDOPS_LOG_TO_STDOUT=1
export ASDOPS_LOG_LEVEL=ERROR
export ASDOPS_LOG_TO_FILE=1
# export ASCEND_SLOG_PRINT_TO_STDOUT=1
# export ASCEND_GLOBAL_LOG_LEVEL=0
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export NPU_MEMORY_FRACTION=0.98
export ATB_WORKSPACE_MEM_ALLOC_ALG_TYPE=3
export ATB_WORKSPACE_MEM_ALLOC_GLOBAL=1
export OMP_NUM_THREADS=12
export HCCL_CONNECT_TIMEOUT=7200
export INF_NAN_MODE_ENABLE=0
export INF_NAN_MODE_FORCE_DISABLE=1
#export ASCEND_LAUNCH_BLOCKING=1
#export ASCEND_HOST_LOG_FILE_NUM=1000
#export ASCEND_GLOBAL_EVENT_ENABLE=1
#export HCCL_ENTRY_LOG_ENABLE=1
#export ASCEND_MODULE_LOG_LEVEL=HCCL=0
# 3. 清理旧日志
\rm -rf core.*
\rm -rf log/node_*.log

# 4. 启动分布式服务
MODEL_PATH=""
MASTER_NODE_ADDR="127.0.0.1:18144"                  # Master 节点地址（需全局一致）
START_PORT=2028                                   # 服务起始端口
DEVICES=(4 5 6 7)                                # NPU 设备列表（选择空闲设备）
LOG_DIR="log"                                      # 日志目录
NNODES=4                                           # 节点数（当前脚本启动 4 个进程）

export HCCL_IF_BASE_PORT=43432  # HCCL 通信基础端口

for (( i=0; i<$NNODES; i++ ))
do
  PORT=$((START_PORT + i))
  DEVICE=$((DEVICES[i]))
  LOG_FILE="$LOG_DIR/node_$i.log"
  ./build/xllm/core/server/xllm \
    --tp_size=4 \
    --model="/export/home/models/wan2_2" \
    --max_memory_utilization=0.98 \
    --backend="dit" \
    --output_shm_size=1024 \
    --devices="npu:$DEVICE" \
    --master_node_addr=$MASTER_NODE_ADDR \
    --nnodes=$NNODES \
    --port $PORT \
    --communication_backend="hccl" \
    --enable_prefix_cache=false \
    --enable_chunked_prefill=false \
    --enable_schedule_overlap=false \
    --use_contiguous_input_buffer=false \
    --dit_debug_print=true \
    --enable-shm=true \
    --node_rank=$i > $LOG_FILE 2>&1 &
done

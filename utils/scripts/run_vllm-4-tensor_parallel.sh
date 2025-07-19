#!/bin/bash
#SBATCH --job-name=vllm-4-tp
#SBATCH --partition=AI4Good_L1_p        
#SBATCH --ntasks=1 # 任务总数
#SBATCH --gpus-per-task=4   # 每个任务4张卡
#SBATCH --cpus-per-task=16   # 每个任务cpu数量
#SBATCH --output=/dev/null  # 丢弃标准输出流
#SBATCH --error=log/vllm_tp_%j.log # 错误日志

# ================= 用户配置区域 =================
MODEL_NAME="/mnt/petrelfs/renqibing/workspace/models/qwen"

PORT_NUM_PER_INSTANCE=20
# 实例端口
MAIN_PORTS=(
    40000
)
# 张量并行度，随gpus-per-task一同变化
TENSOR_PARALLEL_SIZE=4
# 端口转发脚本路径
PORT_FORWARD_SCRIPT="utils/port_forward.py"
# 端口检测脚本路径
PORT_AVALIABLE_SCRIPT="utils/port_avaliable.py"
# ===============================================

# 生成唯一临时文件
PORT_RECORD=$(mktemp -p /tmp port_record_XXXXXX.py)

# 第一阶段：启动所有vllm实例
run_vllm() {
    local port=$1
    srun --gres=gpu:$TENSOR_PARALLEL_SIZE --overlap --ntasks 1 python -m vllm.entrypoints.openai.api_server \
        --model $MODEL_NAME \
        --dtype auto \
        --port $port \
        --trust-remote-code \
        --block-size 32 \
        --gpu-memory-utilization 0.9 \
        --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
        --disable-log-stats &
    echo "已启动vLLM实例，使用端口: $port" >&2
}

for main_port in "${MAIN_PORTS[@]}"; do
    run_vllm $main_port
done

# 第二阶段：安全范围端口分配
declare -a ALL_PORTS
declare -a FORWARD_PIDS

for main_port in "${MAIN_PORTS[@]}"; do
    required_ports=$((PORT_NUM_PER_INSTANCE - 1))
    start_range=$((main_port + 1))
    end_range=$((main_port + PORT_NUM_PER_INSTANCE - 1))
    collected=0
    port_list=()
    
    # 在指定范围内探测
    for (( port=start_range; port<=end_range; port++ )); do
        # 增强型端口检测
        if python $PORT_AVALIABLE_SCRIPT $port
        then
            python $PORT_FORWARD_SCRIPT $port $main_port >> "$PORT_RECORD" 2>&1 &
            FORWARD_PIDS+=($!)
            ALL_PORTS+=($port)
            port_list+=($port)
            ((collected++))
            # echo "[分配成功] $port -> $main_port" >&2
        else
            echo "[端口冲突] $port 已被占用" >&2
            # 向后探测
            ((end_range++))
        fi
    done

    # 严格数量校验
    if [ $collected -ne $required_ports ]; then
        echo "致命错误: 在 $start_range-$end_range 范围内，仅找到 $collected/$required_ports 个可用端口" >&2
        echo "冲突端口可能来自其他实例或外部服务，请检查：" >&2
        ss -tulpn | grep -E ":($(seq -s '|' $start_range $end_range))" >&2
        exit 1
    fi
done

# 格式输出函数
format_list() {
    local arr=("$@")
    printf "[%s]" "$(printf "%s, " "${arr[@]}" | sed 's/, $//')"
}

# 获取公网IP
PUBLIC_IP=$(
    ip -4 addr show eth0 2>/dev/null | grep -oP '(?<=inet\s)\d+(\.\d+){3}' | head -1 ||
    ip -4 addr | grep -v '127.0.0.1' | awk '/inet / {print \$2}' | cut -d/ -f1 | head -1
)

# 信息汇总输出
echo "================ 最终配置 ================" >&2
echo "服务IP: $PUBLIC_IP" >&2
echo "主端口列表: $(format_list "${MAIN_PORTS[@]}")" >&2
echo "转发端口列表: $(format_list "${ALL_PORTS[@]}")" >&2
echo "全部可用端口: $(format_list "${MAIN_PORTS[@]}" "${ALL_PORTS[@]}")" >&2
echo "==========================================" >&2

cleanup() {
    echo "正在清理资源..." >&2
    # 终止所有端口转发进程
    for pid in "${FORWARD_PIDS[@]}"; do
        if kill -0 $pid 2>/dev/null; then
            kill $pid 2>/dev/null
        fi
    done
    
    # 终止所有vLLM进程
    pkill -P $$ 2>/dev/null
    
    # 删除临时文件
    rm -f $PORT_RECORD
    
    echo "资源清理完成" >&2
    exit 0
}

# 注册信号处理
trap cleanup SIGTERM SIGINT SIGHUP EXIT

# 等待所有后台进程
wait

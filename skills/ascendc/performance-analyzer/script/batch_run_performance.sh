# 定义基础目录
BASE_DIR="/home/y00889327/AscendOpGenAgent/output_0410_l2_ziji"

# 定义日志文件路径
LOG_FILE="log_performance.txt"

# 定义锁目录
LOCK_DIR="/tmp/ascend_locks"
mkdir -p "$LOCK_DIR"

# 清空日志
> "$LOG_FILE"

# 初始化设备 ID
device_id=1

# 遍历目录下所有子文件夹
for dir in "$BASE_DIR"/*; do
    if [ -d "$dir" ]; then
        folder_name=$(basename "$dir")
        
        # --- 步骤 1：控制总并发数 (防止系统过载) ---
        while [ $(jobs -r | wc -l) -ge 7 ]; do
            sleep 0.5
        done

        # --- 步骤 2：寻找空闲设备 (防止 NPU 冲突) ---
        # 循环寻找一个既没有脚本在跑，也没有被锁文件占用的设备
        while true; do
            lock_file="$LOCK_DIR/device_${device_id}.lock"
            
            # 检查当前设备是否有锁文件存在 (且被占用)
            # flock -n 尝试加锁，如果失败说明有人拿着锁
            if ! ( set -o noclobber; flock -n 200 ) 200>"$lock_file" 2>/dev/null; then
                # 如果加锁失败（说明设备忙），切换到下一个设备
                device_id=$(( (device_id % 7) + 1 ))
                sleep 0.2 # 稍微等一下再试，避免死循环空转
            else
                # 如果加锁成功，说明设备空闲！
                # 注意：这里不能释放锁，我们要把这个锁“传”给后台任务
                # 但 Shell 很难跨进程传文件描述符。
                # 所以这里我们只做“检查”，真正的互斥靠 ASCEND_RT_VISIBLE_DEVICES 隔离。
                # 为了稳妥，我们打破循环，启动任务。
                break
            fi
        done
        
        echo ">>> 锁定设备 $device_id 并启动算子: $folder_name" | tee -a "$LOG_FILE"
        
        # --- 步骤 3：启动后台任务 ---
        (
            # 在子 Shell 中重新获取锁，确保独占
            lock_file="$LOCK_DIR/device_${device_id}.lock"
            
            # 这里使用 exec 200>"$lock_file" && flock 200 来确保持续持有锁
            exec 200>"$lock_file"
            flock 200
            
            # 设置环境变量
            export ASCEND_RT_VISIBLE_DEVICES=$device_id
            
            # 执行任务
            python3 /home/y00889327/AscendOpGenAgent/skills/ascendc/performance-analyzer/references/performance.py --output_dir "$dir" --output "$dir" 2>&1 | \
            grep -v "tiling struct \[MC2MatmulV3TilingData\] is conflict" | \
            grep -v "tiling struct \[TileInfo\] is conflict" \
            >> "$LOG_FILE"
            
            echo ">>> 算子 $folder_name 在设备 $device_id 上执行完毕" >> "$LOG_FILE"
            
            # 任务结束，锁文件描述符 200 关闭，锁自动释放
        ) &
        
        # 切换设备 ID，为下一个任务做准备
        device_id=$(( (device_id % 7) + 1 ))
    fi
done

# 等待所有任务完成
wait

# 清理
rm -rf "$LOCK_DIR"

echo "所有并行任务已执行完毕，日志已保存至 $LOG_FILE"
#!/bin/bash

CORES_TO_KEEP=16

# 获取从保留数量的下一个核心开始（即 16）到最大核心（127）的所有核心编号
# $(seq 16 127) 会生成 "16 17 18 ... 127" 这样的序列
for cpu in $(seq "$CORES_TO_KEEP" 127); do
    echo "正在下线核心: cpu${cpu}"
    # 使用 sudo tee 向对应的 online 文件写入 0 来下线核心
    # 2>/dev/null 是为了隐藏一些可能的“该核心已经下线”的错误信息
    echo 0 | sudo tee "/sys/devices/system/cpu/cpu${cpu}/online" >/dev/null 2>&1
done

echo "已完成！所有从 cpu16 开始的核心都已下线。"

# 验证一下当前在线的核心
echo "当前在线的核心:"
cat /sys/devices/system/cpu/online
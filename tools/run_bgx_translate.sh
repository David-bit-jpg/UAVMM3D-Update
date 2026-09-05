#!/usr/bin/env bash
# 尺度对齐缓存 mm20c 建好后自动接：RGB 可见度分数 -> 背景翻译（train 全量 + test 全量）-> READY。
# 两个阶段分别由 --stage demo / --stage full 触发，便于先把样例图给用户看，再跑全量：
#   demo：等 mm20c/test/index.pkl 出现 -> mm_vis_score -> 翻译 train 前 40 个候选（打乱后）并出对比图 -> 写 DEMO_DONE
#   full：等 DEMO_DONE -> 翻译 train 剩余候选 -> 翻译 test -> 写 mm20c_bgx/READY
# 断点续跑：translated.npy 记录已翻译帧，重复运行只补没翻的。GPU 只有翻译在用；训练未经用户拍板不启动。
set -u
STAGE=${1:-demo}
SRC=E:/mmcache/mm20c
DST=E:/mmcache/mm20c_bgx
SDPY=/d/SD/venv/Scripts/python.exe
CITYPY=/d/Miniconda3/envs/city/python.exe
export PYTHONUTF8=1 PYTHONPATH=E:/Open3DUAVDet HF_HOME=D:/SD/hf HF_HUB_OFFLINE=1
cd /e/Open3DUAVDet || exit 1
FILT='grep -v -i -E "warn|AutoencoderKL|deprecat|Loading|Hub|local cache"'

wait_for() {  # 文件出现前每 60 s 看一次
    while [ ! -e "$1" ]; do sleep 60; done
}

if [ "$STAGE" = demo ]; then
    wait_for "$SRC/test/index.pkl"
    sleep 30                                          # 等 index 写完整
    if [ ! -e "$SRC/train/vis_score.npy" ] || [ ! -e "$SRC/test/vis_score.npy" ]; then
        echo "== vis_score $(date +%T)"
        $CITYPY tools/mm_vis_score.py --cache "$SRC" --splits train test 2>&1 | grep -v -i warn | tail -4
    fi
    echo "== demo translate $(date +%T)"
    $SDPY tools/sim2real_bg_translate.py --src "$SRC" --split train --dst "$DST" \
        --limit 40 --vis-n 40 --vis-dir "$DST/vis_demo" --batch 4 --steps 20 2>&1 | eval "$FILT"
    touch "$DST/DEMO_DONE"
    echo "== demo done $(date +%T)"
elif [ "$STAGE" = full ]; then
    wait_for "$DST/DEMO_DONE"
    echo "== full translate train $(date +%T)"
    $SDPY tools/sim2real_bg_translate.py --src "$SRC" --split train --dst "$DST" --batch 4 --steps 20 2>&1 | eval "$FILT"
    echo "== full translate test $(date +%T)"
    $SDPY tools/sim2real_bg_translate.py --src "$SRC" --split test --dst "$DST" --batch 4 --steps 20 \
        --vis-n 16 --vis-dir "$DST/vis_test" 2>&1 | eval "$FILT"
    touch "$DST/READY"
    echo "== all done $(date +%T)"
fi

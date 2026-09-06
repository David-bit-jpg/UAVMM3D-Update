#!/usr/bin/env bash
# 交叉贴增广批量生成：挑帧 -> 建不变焦缓存 -> 可见度 -> 抹除底图 -> SD 翻译背景 -> 坐标自检 -> 贴 N 张 -> 打包成训练缓存 -> 查看器
# 用法： bash tools/run_paste_aug_batch.sh [N=3000] [OUT=E:/data_collect/aug_paste_v1]
# 每一步有产物就跳过，可反复运行续跑。日志在 $OUT/logs/。
set -u
N=${1:-3000}
OUT=${2:-E:/data_collect/aug_paste_v1}
CACHE=E:/mmcache/paste_src
CITYPY=/d/Miniconda3/envs/city/python.exe
SDPY=/d/SD/venv/Scripts/python.exe
export PYTHONUTF8=1 PYTHONPATH=E:/Open3DUAVDet HF_HOME=D:/SD/hf HF_HUB_OFFLINE=1
cd /e/Open3DUAVDet || exit 1
mkdir -p "$OUT/logs"
LOG=$OUT/logs/driver.log
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
FILT='grep -v -i -E "warn|AutoencoderKL|deprecat|Loading|Hub|local cache"'

say "== 1 挑帧"
[ -f "$OUT/frames.txt" ] || $CITYPY tools/select_paste_frames.py --out "$OUT/frames.txt" --per-class 250 --n-bg 1200 2>&1 | tee -a "$LOG"

say "== 2 建缓存（--crop-to-mav6d，不变焦）"
if [ ! -f "$CACHE/train/index.pkl" ]; then
  (cd tools && $CITYPY build_mm_cache.py --list "$OUT/frames.txt" --split train --root E:/data_collect --out "$CACHE" --crop-to-mav6d --every 1 --workers 8) 2>&1 | grep -v -i warn | tail -3 | tee -a "$LOG"
fi

say "== 3 可见度分数"
[ -f "$CACHE/train/vis_score.npy" ] || (cd tools && $CITYPY mm_vis_score.py --cache "$CACHE" --splits train) 2>&1 | tail -1 | tee -a "$LOG"

say "== 4 抹除底图"
[ -f "${CACHE}_erased/train/rgb.npy" ] || $CITYPY tools/mm_paste_aug.py --cache "$CACHE" --erased "${CACHE}_erased" --make-plates 2>&1 | grep -v -i warn | tail -1 | tee -a "$LOG"

say "== 5 SD 翻译背景"
[ -f "${CACHE}_plates/train/READY" ] || $SDPY tools/sim2real_bg_translate.py --src "${CACHE}_erased" --split train --dst "${CACHE}_plates" --no-paste --no-erase --min-vis 0 --max-targets 99 --batch 4 --steps 20 2>&1 | eval "$FILT" | tail -2 | tee -a "$LOG"

say "== 6 坐标链路自检"
$CITYPY tools/mm_paste_aug.py --cache "$CACHE" --erased "${CACHE}_erased" --plates "${CACHE}_plates" --selftest 10 --out "$OUT/logs/selftest" 2>&1 | grep -v -i warn | tail -12 | tee -a "$LOG"

say "== 7 交叉贴 $N 张"
$CITYPY tools/mm_paste_aug.py --cache "$CACHE" --erased "${CACHE}_erased" --plates "${CACHE}_plates" --n "$N" --no-sheets \
    --range-ref docs/results/mav6d_size_px_range_m.npy --tone 0.6 --seed 2026 --out "$OUT/samples" 2>&1 | grep -v -i warn | tail -12 | tee -a "$LOG"

say "== 8 打包成训练缓存"
$CITYPY tools/pack_paste_cache.py --samples "$OUT/samples" --out "$OUT/cache" \
    --note "run_paste_aug_batch.sh N=$N: select(per-class 250, bg 1200) -> build_mm_cache --crop-to-mav6d -> sim2real_bg_translate 0.45 neutral -> mm_paste_aug --range-ref mav6d --tone 0.6 --seed 2026" 2>&1 | tail -2 | tee -a "$LOG"

say "== 9 查看器（前 200 个）"
$CITYPY tools/make_paste_viewer.py --dir "$OUT/samples" --out "$OUT/viewer" --cache "$CACHE" --limit 200 2>&1 | tail -1 | tee -a "$LOG"
say "== ALL DONE -> $OUT"

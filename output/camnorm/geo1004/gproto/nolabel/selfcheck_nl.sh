#!/bin/bash
# 11-10 无标签链自查：列出链上每个脚本，grep 真实训练集真值字段（boxes9d / gt / cls / names / qualified / center_inside / 序列前缀分组 / index.pkl / sealed）。
# 期望命中只有：(1) index_pseudo_<arm>.pkl 的 boxes9d（= 伪框，不是真值）；(2) w_sim.py 的仿真真值；(3) fuse_eval_nl.py / chain_eval_m_nl.ang_metrics 的最终打分
# （cmpeval/<split>_meta.npz 封存的 val/test 真值）；(4) make_train_nogt_index.py 打开 index.pkl 只取 seq/frame/K_in；(5) 置空的 gt 字段。
NL=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); R=$(cd "$NL/../../../../.." && pwd); RW=$(cd "$R" && (pwd -W 2>/dev/null || pwd))   # 仓库根（RW = Windows 写法，给 python 参数用）
G=$R/output/camnorm/geo1004
OUT=$NL/selfcheck_nl.txt
FILES="$NL/run_chain_nl.sh $NL/make_train_nogt_index.py $NL/eval_dump_nl.py $NL/dump_train_kz_nl.py $NL/xpaste_bank_sf2_nl.py $NL/extract_nbm_nl.py
$NL/secondstage_v2_nl.py $NL/adabn_all_nl.py $NL/kps_nl.py $G/gproto/wf23/fusetrain/recut_mirror_t.py $NL/w_sim.py $NL/final_unc_nl.py
$NL/ztrain_nl.py $NL/build_targets_nl.py $NL/chain_eval_m_nl.py $NL/fuse_eval_nl.py $NL/make_nolabel_pseudo.py"
PAT="boxes9d|\['gt'\]|\"gt\"|'gt'|\bgt\b|\bcls\b|'names'|qualified|center_inside|index\.pkl|sealed|split\('/'\)|with_gt=True|zgt|n_gt"
{
echo "# 无标签链自查 $(date '+%m-%d %H:%M')"
echo "## 链上脚本（${FILES//$'\n'/ }）"
for f in $FILES; do
  echo; echo "=== $f"
  grep -nE "$PAT" "$f" | grep -v '^\s*#' | sed 's/^/  /' || true
done
} > $OUT
cat $OUT

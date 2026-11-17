# [review1113 ourdet_baseproto] shared settings. combo: in2mav | out2antifs
export PYTHONUTF8=1 PYTHONPATH=E:/Open3DUAVDet TORCH_HOME=E:/torch_home UAV_PNP_EPNP_FALLBACK=1
H=/e/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/ourdet_baseproto; HW=E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/ourdet_baseproto
PY=D:/Miniconda3/envs/city/python.exe; SLOT="bash /e/Open3DUAVDet/output/camnorm/geo1004/gproto/gpu_slot.sh"
MR=D:/S2MUAV3D/ourdet_baseproto_models/models/uavdet_3d/camnorm/ourdet_baseproto
case $C in
  in2mav)     export SX_SRC=in SX_TGT=mav; DET0=E:/Open3DUAVDet/output/release_20260920/ramp_s0.pth; SPLIT=real_test
              TRAIN_COPY=F:/ourdet_baseproto_cache/mav6d_cn_obj ;;
  out2antifs) export SX_SRC=out SX_TGT=antifs; DET0=E:/Open3DUAVDet/output/models/uavdet_3d/camnorm/sim_out_r34_geo_ramp_out/DET_s0/ckpt/checkpoint_epoch_5.pth; SPLIT=antifs_test
              TRAIN_COPY=D:/S2MUAV3D/ourdet_baseproto_cache/outdoor_real_fs ;;
  *) echo "bad combo $C"; exit 2 ;;
esac
export SX_TEACH=full
say() { echo "$(date '+%m-%d %H:%M:%S') [$C] $1" | tee -a $H/logs/run.log; }

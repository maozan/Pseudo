tport=53907
ngpu=2
ROOT=.
# 是单机多卡的方式

#CUDA_VISIBLE_DEVICES=0,3, 搭配ngpu=2使用，即使用0,3号两张显卡
CUDA_VISIBLE_DEVICES=0,3, \
python -m torch.distributed.launch \
    --nproc_per_node=${ngpu} \
    --node_rank=0 \
    --master_port=${tport} \
    $ROOT/train_semi.py \
    --config=$ROOT/exps/compare/semi_voc_blender/r50_662/config_semi.yaml --seed 2

    # --config=$ROOT/exps/mrun_vocs/cutmix/config_semi.yaml --seed 2
    # --config=$ROOT/exps/mrun_vocs/cutadap/config_semi.yaml --seed 2


# ---- -----
    # --config=$ROOT/exps/zrun_vocs/voc_semi_fine92/config_semi.yaml --seed 2 --port ${tport}
    # --config=$ROOT/exps/zrun_vocs/voc_semi_fine183/config_semi.yaml --seed 2 --port ${tport}
    # --config=$ROOT/exps/zrun_vocs/voc_semi_fine366/config_semi.yaml --seed 2 --port ${tport}
    # --config=$ROOT/exps/zrun_vocs/voc_semi_fine732/config_semi.yaml --seed 2 --port ${tport}
    # --config=$ROOT/exps/zrun_vocs/voc_semi_fine1464/config_semi.yaml --seed 2 --port ${tport}

    # --config=$ROOT/exps/zrun_vocs/voc_semi662/config_semi.yaml --seed 2 --port ${tport}
    # --config=$ROOT/exps/zrun_vocs/voc_semi1323/config_semi.yaml --seed 2 --port ${tport}
    # --config=$ROOT/exps/zrun_vocs/voc_semi2646/config_semi.yaml --seed 2 --port ${tport}
    # --config=$ROOT/exps/zrun_vocs/voc_semi5291/config_semi.yaml --seed 2 --port ${tport}

    # --config=$ROOT/exps/zrun_vocs/r50_voc_semi662/config_semi.yaml --seed 2 --port ${tport}
    # --config=$ROOT/exps/zrun_vocs/r50_voc_semi1323/config_semi.yaml --seed 2 --port ${tport}
    # --config=$ROOT/exps/zrun_vocs/r50_voc_semi2646/config_semi.yaml --seed 2 --port ${tport}
    # --config=$ROOT/exps/zrun_vocs/r50_voc_semi5291/config_semi.yaml --seed 2 --port ${tport}

# ---- ---- u2pl
    # --config=$ROOT/exps/zrun_vocs_u2pl/voc_semi662/config_semi.yaml --seed 2 --port ${tport}
    # --config=$ROOT/exps/zrun_vocs_u2pl/voc_semi1323/config_semi.yaml --seed 2 --port ${tport}
    # --config=$ROOT/exps/zrun_vocs_u2pl/voc_semi2646/config_semi.yaml --seed 2 --port ${tport}
    # --config=$ROOT/exps/zrun_vocs_u2pl/voc_semi5291/config_semi.yaml --seed 2 --port ${tport}

    # --config=$ROOT/exps/zrun_vocs_u2pl/r50_voc_semi662/config_semi.yaml --seed 2 --port ${tport}
    # --config=$ROOT/exps/zrun_vocs_u2pl/r50_voc_semi1323/config_semi.yaml --seed 2 --port ${tport}
    # --config=$ROOT/exps/zrun_vocs_u2pl/r50_voc_semi2646/config_semi.yaml --seed 2 --port ${tport}
    # --config=$ROOT/exps/zrun_vocs_u2pl/r50_voc_semi5291/config_semi.yaml --seed 2 --port ${tport}
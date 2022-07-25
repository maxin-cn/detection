python tools/test.py \
configs/jiejing/cascade_s50_fpn_label_smooth_3x-inference.py \
/mnt/dolphinfs/hdd_pool/docker/user/hadoop-basecv/maxin/work/roadseg/chekpoint/epoch_10.pth \
--eval bbox segm --options="classwise=True"


python tools/test.py \
configs/jiejing/cascade_s50_fpn_label_smooth_3x-inference-fanet.py \
/mnt/dolphinfs/hdd_pool/docker/user/hadoop-basecv/maxin/work/roadseg/epoch_10.pth \
--eval bbox segm --options="classwise=True"


python tools/test.py \
configs/jiejing/cascade_s50_fpn_label_smooth_3x-inference.py \
/mnt/dolphinfs/hdd_pool/docker/user/hadoop-basecv/maxin/work/roadseg/work_dirs/cascade_s50_fpn_label_smooth_3x-train/epoch_12.pth \
--eval bbox segm --options="classwise=True"


python tools/test.py \
configs/jiejing/cascade_s50_fpn_label_smooth_3x-inference-fanet.py \
/mnt/dolphinfs/hdd_pool/docker/user/hadoop-basecv/maxin/work/roadseg/work_dirs/cascade_s50_fpn_label_smooth_3x-train-fanet/epoch_12.pth \
--eval bbox segm --options="classwise=True"


python tools/test.py \
configs/jiejing/cascade_s50_fpn_label_smooth_3x-inference-fanet.py \
/mnt/dolphinfs/hdd_pool/docker/user/hadoop-basecv/maxin/work/roadseg/work_dirs/cascade_s50_fpn_label_smooth_3x-train-fanet/epoch_12.pth \
--eval bbox segm --options="classwise=True"


python tools/test.py \
configs/jiejing/cascade_s50_fpn_label_smooth_3x-inference.py \
/mnt/dolphinfs/hdd_pool/docker/user/hadoop-basecv/maxin/work/roadseg/work_dirs/cascade_s50_fpn_label_smooth_reload_6x/epoch_12.pth \
--eval bbox segm --options="classwise=True"

python tools/test.py \
configs/jiejing/cascade_s50_fpn_label_smooth_3x-inference-fanet.py \
/mnt/dolphinfs/hdd_pool/docker/user/hadoop-basecv/maxin/work/roadseg/work_dirs/cascade_s50_fpn_label_smooth_reload_6x_fanet/epoch_10.pth \
--eval bbox segm --options="classwise=True"

python tools/test.py \
configs/jiejing/cascade_s50_fpn_label_smooth_3x-inference-fanet.py \
/mnt/dolphinfs/hdd_pool/docker/user/hadoop-basecv/maxin/work/roadseg/work_dirs/cascade_s50_fpn_label_smooth_reload_6x_fanet_b1/epoch_6.pth \
--eval bbox segm --options="classwise=True"


python tools/test.py \
configs/jiejing/cascade_s50_fpn_label_smooth_3x-inference-fanet.py \
/mnt/dolphinfs/hdd_pool/docker/user/hadoop-basecv/maxin/work/roadseg/work_dirs/cascade_s50_fpn_label_smooth_reload_6x_fanet_b0/epoch_12.pth \
--eval bbox segm --options="classwise=True"



# analog_autoformer
to create the envirement: 

conda env create -f environment.yaml
conda activate analog-autoformer-lightning

and then test : 
from aihwkit_lightning.nn import AnalogLinear
from aihwkit_lightning.simulator.configs import TorchInferenceRPUConfig
print("AIHWKIT-Lightning ready")

to train : 
run : 
python training_script.py \
    --cfg AutoFormer/experiments/supernet/supernet-T.yaml \
    --digital-ckpt digital_ckpts/supernet-tiny.pth \
    --analog-config configs/analog-supernet-T.yaml \
    --data-path /home/douaa/.cache/kagglehub/datasets/ifigotin/imagenetmini-1000/versions/1/imagenet-mini \
    --batch-size 64 \
    --epochs 100 \
    --output_dir ./analog-ckpts \
    --gp --change_qkv --relative_position

to train tiny : 

python training_script.py \
    --cfg AutoFormer/experiments/supernet/supernet-T.yaml \
    --digital-ckpt digital_ckpts/supernet-tiny.pth \
    --analog-config configs/analog-supernet-T.yaml \
    --data-path /home/douaa/.cache/kagglehub/datasets/ifigotin/imagenetmini-1000/versions/1/imagenet-mini \
    --batch-size 64 \
    --epochs 100 \
    --output_dir ./analog-ckpts \
    --gp --change_qkv --relative_position
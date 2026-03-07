# analog_autoformer
to create the envirement: 

conda env create -f environment.yaml
conda activate analog-autoformer-lightning



to train : 
run : 
python training_script.py \
    --cfg AutoFormer/experiments/supernet/supernet-T.yaml \
    --digital-ckpt digital_ckpts/supernet-tiny.pth \
    --analog-config configs/analog-supernet-T.yaml \
    --data-path path-to-imagenet \
    --batch-size 64 \
    --epochs 100 \
    --output_dir ./analog-ckpts \
    --gp --change_qkv --relative_position


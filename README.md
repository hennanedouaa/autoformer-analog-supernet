# analog_autoformer
1-To create the envirement: 


```buildoutcfg
conda env create -f environment.yaml
conda activate analog-autoformer-supernet
```

2-To run training script:   
for tiny:   
```buildoutcfg
python  training_script.py \
  --cfg AutoFormer/experiments/supernet/supernet-T.yaml  \
 --digital-ckpt digital_ckpts/supernet-tiny.pth \
  --analog-config configs/analog-supernet-T.yaml \
  --data-path path-to-imagenet  \
 --batch-size 64 \
  --epochs 100 \
  --output_dir ./analog-ckpts \
  --gp --change_qkv --relative_position
```


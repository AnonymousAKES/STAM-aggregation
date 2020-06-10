# Train Transformer with STAM aggregation
## Training with IWSLT14 German to English

1. Data preparation. Follow the instructions in [this link](https://github.com/pytorch/fairseq/blob/v0.9.0/examples/translation/README.md) to download and preprocess IWSLT14 De-En dataset.

2. Model training. The following command trains a model with 32 heads, 64 head dimension, 4096 fc dimension and 256 model dimension.  We set alpha as 1/sqrt(C/4). 

```
C=32
d_h=64
d_fc=4096
d=256
alpha=0.35

CUDA_VISIBLE_DEVICES=0 python train.py     data-bin/iwslt14.tokenized.de-en     --arch transformer_iwslt_de_en \
--optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0     --lr 1e-3 --lr-scheduler inverse_sqrt  --warmup-updates 4000 \
--dropout 0.3 --weight-decay 0.0001 --attention-dropout 0.3 --relu-dropout 0.3 --head_dropout 0.3 --no-epoch-checkpoints \
--criterion label_smoothed_cross_entropy --label-smoothing 0.1 --share-all-embeddings --max-tokens 4096 \
--encoder-attention-heads ${C} --decoder-attention-heads ${C} --encoder-embed-dim ${d} --decoder-embed-dim ${d} \
--alpha ${alpha} --fc_alpha ${alpha} --encoder-ffn-embed-dim ${d_fc} --decoder-ffn-embed-dim ${d_fc}  --attn_head_dim ${d_h}  --no-progress-bar \
--save-dir heads${C}_headdim${d_h}_fcdim${d_fc}_modeldim${d}_alpha${alpha}
```

3. Inference

```
 CUDA_VISIBLE_DEVICES=0 python generate.py data-bin/iwslt14.tokenized.de-en \
    --path ckpt_dir/checkpoint_best.pt \
    --batch-size 256 --beam 5  --remove-bpe --quiet  
```

## Training with WMT16  English to German

1. Data preparation. Follow the instructions in [this link](https://github.com/pytorch/fairseq/blob/v0.9.0/examples/scaling_nmt/README.md) to download and preprocess WMT16 En-De dataset.

2. Model training.
For WMT16 En-De, we set alpha as 1/sqrt(C/8). We use a single node with 8 V100 cards. The following command trains a model with 36 heads, 64 head dimension, 9216 model dimension and 512 model dimension. 

```
C=36
d_h=64
d_fc=9216
d=512
alpha=0.471

python train.py     data-bin/wmt16_en_de_bpe32k     --arch transformer_vaswani_wmt_en_de_big --share-all-embeddings  --warmup-init-lr 1e-07 \
--optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0   --lr 0.001000 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
--encoder-embed-dim ${d} --decoder-embed-dim ${d} --encoder-ffn-embed-dim ${d_fc}    --decoder-ffn-embed-dim ${d_fc} --fp16 \
 -s en -t de --dropout 0.3 --head_dropout 0.1 --attention-dropout 0.1 --relu-dropout 0.1 --weight-decay 0.    --criterion label_smoothed_cross_entropy --label-smoothing 0.1    --max-tokens 4096 --update-freq 4  --fp16  \
--encoder-attention-heads ${C} --decoder-attention-heads ${C} --attn_head_dim ${d_h}     --alpha ${alpha} --fc_alpha ${alpha}  \ 
--ddp-backend=no_c10d --no-progress-bar --keep-last-epochs 20 --save-dir checkpoints/wmt_heads${C}_headdim${d_h}_fcdim${d_fc}_modeldim${d}_alpha${alpha}
```

We note that the '--fp16' option requires a GPU with Volta architecture.

3. Inference. First average the last 5 checkpoints.

```
python scripts/average_checkpoints \
    --inputs /path/to/checkpoints \
    --num-epoch-checkpoints 5 \
    --output checkpoint_avergae_last5.pt
```

Then follow the instructions in [this link](https://github.com/pytorch/fairseq/blob/v0.9.0/examples/scaling_nmt/README.md) to evaluate BLEU score.

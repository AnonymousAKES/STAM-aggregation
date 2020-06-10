import os
import math
import numpy as np
i =0
attn_dim = 1024
embed_dim = 1024 # 256
ffn_dim = 2048 # 512
# heads = 8, 12, 16, 32
# baseline  4/512/1024 
# multi-branches:  8/256/1024  12/256/1536  16/256/2048    32/256/4096

for heads in [4, 256]:#,[64,128,256]:#
    for dropout in [0.]:
        for act_drop in [0.]:
            for attn_drop in [0.,]:
                for ptauscale in [1.]:
                    for lr in [5e-4]:
                        for head_dim in [64]:
                            # ffn_dim = 512 * (heads//4)
                            # alpha = 1/math.sqrt(heads//4)
                            head_dim = attn_dim//heads
                            alpha = 1.
                            head_drop = act_drop
                            rand_seed = np.random.randint(100000)
                            epoch = 20

                            sess = 'multi-head_big_wd0iwlst_%d_%d_alpha%.2f_embed%d_ffndim%d_lr%.4f_dropout%.2f_attndrop%.2f_actdrop%.2f_headdrop%.2f_rand_seed%d'%(heads, head_dim, alpha, embed_dim, ffn_dim,lr, dropout, attn_drop, act_drop, head_drop,rand_seed)

                            # cmd = 'CUDA_VISIBLE_DEVICES=%d, python train.py     data-bin/iwslt14.tokenized.de-en     --arch transformer_iwslt_de_en --share-all-embeddings     --optimizer adam --adam-betas \'(0.9, 0.98)\' --clip-norm 0.0 \
                            # --lr %f --lr-scheduler inverse_sqrt --warmup-updates 4000 --encoder-embed-dim %d --decoder-embed-dim %d --encoder-ffn-embed-dim %d --decoder-ffn-embed-dim %d --attention-dropout %f --relu-dropout %f --head_dropout %f --dropout %f --weight-decay 0.000     --criterion label_smoothed_cross_entropy --label-smoothing 0.1     --max-tokens 4096 \
                            # --encoder-attention-heads %d --decoder-attention-heads %d --attn_head_dim %d --ptau_scale %f --save-dir checkpoints/final/%s --no-progress-bar --no-epoch-checkpoints --eval-bleu  --eval-bleu-args \'{\"beam\": 5, \"max_len_a\": 1.2, \"max_len_b\": 10}\'    --eval-bleu-detok moses     --eval-bleu-remove-bpe     --eval-bleu-print-samples\
                            # --best-checkpoint-metric bleu  --maximize-best-checkpoint-metric --alpha %f --seed %d --down_init %f --max-epoch %d >> logs/%s.txt \
                            #         &'%(i%4,  lr, embed_dim, embed_dim, ffn_dim, ffn_dim, attn_drop, act_drop, head_drop, dropout, heads, heads, head_dim, -1.,  sess, alpha, rand_seed, down_init, epoch, sess)
                            cmd = 'CUDA_VISIBLE_DEVICES=%d,%d python train.py     data-bin/iwslt14.tokenized.de-en     --arch transformer_iwslt_de_en --share-all-embeddings     --optimizer adam --adam-betas \'(0.9, 0.98)\' --clip-norm 0.0 \
                            --lr %f --lr-scheduler inverse_sqrt --warmup-updates 4000 --encoder-embed-dim %d --decoder-embed-dim %d --encoder-ffn-embed-dim %d --decoder-ffn-embed-dim %d --attention-dropout %f --relu-dropout %f --head_dropout %f --dropout %f --weight-decay 0.000     --criterion label_smoothed_cross_entropy --label-smoothing 0.1     --max-tokens 2048 \
                            --encoder-attention-heads %d --decoder-attention-heads %d --encoder-normalize-before --decoder-normalize-before --attn_head_dim %d  --save-dir checkpoints/%s --no-progress-bar --no-epoch-checkpoints \
                             --alpha %f --seed %d  --max-epoch %d >> logs/%s.txt\
                                &'%(i%4, i%4+1, lr, embed_dim, embed_dim, ffn_dim, ffn_dim, attn_drop, act_drop, head_drop, dropout, heads, heads, head_dim,   sess, alpha, rand_seed, epoch, sess)

                            i+=2
                            os.system(cmd)
            #os.system(cmd1)
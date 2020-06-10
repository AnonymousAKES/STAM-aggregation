# ResNeXt on CIFAR

## Train ResNeXt with STAM aggregation.

The following command trains ResNeXt8x64 on CIFAR10 with default uniform alpha.

`python cifar_train.py --num_parallel 8 --sess ResNeXt8x64_STAM`

You can also use non-uniform alpha and/or CIFAR100.

`python cifar_train.py --num_parallel 8 --no_uniform --c100 --sess ResNeXt8x64_STAM_nonuniform_c100`

Train ResNeXt16x64 with STAM aggregation.

`python cifar_train.py --num_parallel 16 --sess ResNeXt16x64_STAM`

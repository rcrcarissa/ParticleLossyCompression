An Error-Bounded Lossy Compression Method with Bit-Adaptive Quantization for Particle Data
=====
(C) 2024 by Congrong Ren. See LICENSE in top-level directory.

## 3rd party libraries/tools

[Zstandard](https://facebook.github.io/zstd/).

## Testing Examples

Please use the executable 'comp_decomp' command to do the compression & decompression. Parameters are listed below.

Parameter | Explanation | Options
--- | --- | --- 
-ds | Specify the dataset name. | N/A
-dim | Specify the dimension. | 2 or 3
-reb | Specify the relative error bound. | N/A
-r | Specify the maximum number of particles in a leaf node. | N/A
-f/-d | Specify the precision. | -f or -d

For example:
```
$ comp_decomp -ds Nyx -dim 3 -reb 1e-9 -r 50 -d
```

## Citation

Please including the following citation if you use the code:

* C. Ren, S. Di, L. Zhang, K. Zhao, and H. Guo, "[An Error-Bounded Lossy Compression Method with Bit-Adaptive Quantization for Particle Data](https://arxiv.org/abs/2404.02826)," arXiv preprint arXiv:2404.02826, 2024.

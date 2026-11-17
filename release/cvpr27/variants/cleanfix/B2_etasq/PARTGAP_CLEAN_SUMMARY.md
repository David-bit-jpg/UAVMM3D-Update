# cleanfix B2: part-level domain gap on CLEAN real crops (srcx/in2mav det1 train crops); sim = partgap sim_samples (885)

seeds [0, 1, 2]; n_sim 885, n_real 2000

| layer | pool | AUC mean | eta2 | Frechet |
|---|---|---|---|---|
| layer2 | win | 1.000 | 0.285 | 1.403 |
| layer2 | ctr | 1.000 | 0.156 | 0.770 |
| layer2 | motor | 0.997 | 0.064 | 0.356 |
| layer2 | nose | 0.965 | 0.021 | 0.196 |
| layer2 | tail | 0.960 | 0.020 | 0.174 |
| layer2 | top | 0.978 | 0.025 | 0.198 |
| layer2 | bottom | 0.965 | 0.022 | 0.189 |
| layer2 | body8 | 0.998 | 0.061 | 0.356 |
| layer3 | win | 1.000 | 0.226 | 1.168 |
| layer3 | ctr | 1.000 | 0.079 | 0.552 |
| layer3 | motor | 0.998 | 0.048 | 0.418 |
| layer3 | nose | 0.987 | 0.026 | 0.323 |
| layer3 | tail | 0.980 | 0.025 | 0.324 |
| layer3 | top | 0.992 | 0.033 | 0.393 |
| layer3 | bottom | 0.989 | 0.034 | 0.429 |
| layer3 | body8 | 0.999 | 0.053 | 0.456 |
| layer4 | win | 1.000 | 0.160 | 0.944 |
| layer4 | ctr | 0.998 | 0.065 | 0.635 |
| layer4 | motor | 0.997 | 0.064 | 0.598 |
| layer4 | nose | 0.990 | 0.038 | 0.513 |
| layer4 | tail | 0.986 | 0.038 | 0.501 |
| layer4 | top | 0.987 | 0.035 | 0.497 |
| layer4 | bottom | 0.991 | 0.030 | 0.468 |
| layer4 | body8 | 0.995 | 0.060 | 0.585 |
| head | win | 1.000 | 0.160 | 0.882 |
| head | ctr | 0.999 | 0.085 | 0.581 |
| head | motor | 0.994 | 0.048 | 0.496 |
| head | nose | 0.970 | 0.024 | 0.363 |
| head | tail | 0.966 | 0.024 | 0.357 |
| head | top | 0.986 | 0.024 | 0.421 |
| head | bottom | 0.987 | 0.023 | 0.446 |
| head | body8 | 0.997 | 0.053 | 0.522 |

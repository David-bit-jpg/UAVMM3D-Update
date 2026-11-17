# cleanfix B2: part-level domain gap on CLEAN real crops (srcx/in2mav det1 train crops); sim = partgap sim_samples (885)

seeds [0]; n_sim 12, n_real 24

| layer | pool | AUC mean | eta2 | Frechet |
|---|---|---|---|---|
| layer2 | win | 0.150 | 0.248 | 1.307 |
| layer2 | ctr | 0.250 | 0.151 | 0.948 |
| layer2 | motor | 0.430 | 0.050 | 0.756 |
| layer2 | nose | 0.643 | 0.035 | 0.745 |
| layer2 | tail | 0.517 | 0.046 | 0.774 |
| layer2 | top | 0.373 | 0.053 | 0.781 |
| layer2 | bottom | 0.557 | 0.044 | 0.732 |
| layer2 | body8 | 0.410 | 0.054 | 0.776 |
| layer3 | win | 0.247 | 0.238 | 1.281 |
| layer3 | ctr | 0.283 | 0.106 | 0.925 |
| layer3 | motor | 0.470 | 0.066 | 0.892 |
| layer3 | nose | 0.460 | 0.064 | 0.920 |
| layer3 | tail | 0.577 | 0.050 | 0.903 |
| layer3 | top | 0.517 | 0.058 | 0.880 |
| layer3 | bottom | 0.373 | 0.059 | 0.903 |
| layer3 | body8 | 0.420 | 0.076 | 0.878 |
| layer4 | win | 0.403 | 0.149 | 0.913 |
| layer4 | ctr | 0.493 | 0.079 | 0.870 |
| layer4 | motor | 0.540 | 0.081 | 0.905 |
| layer4 | nose | 0.390 | 0.069 | 0.901 |
| layer4 | tail | 0.433 | 0.077 | 0.928 |
| layer4 | top | 0.333 | 0.049 | 0.834 |
| layer4 | bottom | 0.460 | 0.043 | 0.835 |
| layer4 | body8 | 0.440 | 0.080 | 0.900 |
| head | win | 0.520 | 0.082 | 0.386 |
| head | ctr | 0.587 | 0.063 | 0.687 |
| head | motor | 0.333 | 0.053 | 0.881 |
| head | nose | 0.363 | 0.049 | 0.911 |
| head | tail | 0.347 | 0.052 | 0.914 |
| head | top | 0.500 | 0.043 | 0.878 |
| head | bottom | 0.367 | 0.043 | 0.919 |
| head | body8 | 0.220 | 0.063 | 0.889 |

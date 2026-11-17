# Artifacts

Prediction files behind every reported number (included under `preds/`; run `REPRO_PRED_ROOT=<this folder>/preds python rescore_all.py`).
The scorer (`D:/UAV3D_baselines_pack/scripts/score.py`, copy in `baselines/advisor_pack/00_UAV3D_baselines_pack/scripts/`) and the sealed test GT metas are read from their original paths:

- GT mav: `D:/UAV3D_baselines_pack/data/gt/real_test_meta.npz` sha256 0bc3cdd174cf5b70f626f5a254947009f61dc867ac9a07027b2ee6ec3fea3f41
- GT anti: `D:/S2MUAV3D/real_outdoor_antiuav3d_fs/labels/sealed/antifs_test_meta.npz` sha256 dd05225e116f41ad8aa2ba0ff908ef1e4ccab03d48847cef85ceb08ff9132f45

## Prediction files

| path | size | sha256 | status |
|---|---|---|---|
| `D:/CVPR27_UAV/figures/data/pl_quality_mav6d.csv` | 2508782 | 143a59adf916a2300619efc8ff625ff8ee89056d3380ddf6821762504db37ab0 | included at repro/preds/D/CVPR27_UAV/figures/data/pl_quality_mav6d.csv |
| `D:/advisor_results_1117b/runs/gupnet_in_kdc_mav_s0/pred_real_test.pkl` | 4366852 | c3e9432164183fa53759a5a9b33b1dff931eb7c6d89088587b729727443ba534 | included at repro/preds/D/advisor_results_1117b/runs/gupnet_in_kdc_mav_s0/pred_real_test.pkl |
| `D:/advisor_results_1117b/runs/gupnet_in_kdc_mav_s1/pred_real_test.pkl` | 4366852 | 1ca7080c8c3bd8f434814bdf623676ae4e83f16ad63257c01dac8e63a9f5e0d4 | included at repro/preds/D/advisor_results_1117b/runs/gupnet_in_kdc_mav_s1/pred_real_test.pkl |
| `D:/advisor_results_1117b/runs/gupnet_in_kdc_mav_s2/pred_real_test.pkl` | 4366852 | 21eea0f63dec55ca0f50f9bc87add9974174a188b9245eb1d0f73bf0323e06b4 | included at repro/preds/D/advisor_results_1117b/runs/gupnet_in_kdc_mav_s2/pred_real_test.pkl |
| `D:/advisor_results_1117b/runs/gupnet_in_sim_s1/pred_real_test.pkl` | 4366852 | 9c2aad408c796acfa5df365828660b7e81223575c1902b6c81a485132239ae6c | included at repro/preds/D/advisor_results_1117b/runs/gupnet_in_sim_s1/pred_real_test.pkl |
| `D:/advisor_results_1117b/runs/gupnet_in_sim_s2/pred_real_test.pkl` | 4366852 | 2668d15f278145c02be1095becdddbdae95ab700e68835db1c1a3349b9dc7910 | included at repro/preds/D/advisor_results_1117b/runs/gupnet_in_sim_s2/pred_real_test.pkl |
| `D:/advisor_results_1117b/runs/gupnet_in_st_mav_s0/pred_real_test.pkl` | 4366852 | 8d3a6adcfb6127fb21615f0834f3f03d67f0a5a32bd0402661ac7822a5c7d12d | included at repro/preds/D/advisor_results_1117b/runs/gupnet_in_st_mav_s0/pred_real_test.pkl |
| `D:/advisor_results_1117b/runs/gupnet_out_dann_anti_s0/pred_outdoor_test.pkl` | 2343005 | 60db89e467db1723c603335b1b6be09de976b005a04021534d8c051014f779d2 | included at repro/preds/D/advisor_results_1117b/runs/gupnet_out_dann_anti_s0/pred_outdoor_test.pkl |
| `D:/advisor_results_1117b/runs/gupnet_out_kd_anti_s0/pred_outdoor_test.pkl` | 2343005 | a051b5b799a30cf7f6ad151fe8d208b10721e1121e99e610ccf0aa0c02e13eaf | included at repro/preds/D/advisor_results_1117b/runs/gupnet_out_kd_anti_s0/pred_outdoor_test.pkl |
| `D:/advisor_results_1117b/runs/gupnet_out_kd_anti_s1/pred_outdoor_test.pkl` | 2343005 | 36a9f62b385a5e77f05c9a2e41434301c5b4f7b7af4ab8512a505c94425dcead | included at repro/preds/D/advisor_results_1117b/runs/gupnet_out_kd_anti_s1/pred_outdoor_test.pkl |
| `D:/advisor_results_1117b/runs/gupnet_out_kd_anti_s2/pred_outdoor_test.pkl` | 2343005 | d3f25b5063d2756cab0640d554b26eb90d07be6ea9ba161b70417cc82d7e9689 | included at repro/preds/D/advisor_results_1117b/runs/gupnet_out_kd_anti_s2/pred_outdoor_test.pkl |
| `D:/advisor_results_1117b/runs/gupnet_out_sim_s0/pred_outdoor_test.pkl` | 2343005 | ce2150b2260dbe57a17a5fb7ba649adb2624e8e3bf9c89c425d9381482d194a8 | included at repro/preds/D/advisor_results_1117b/runs/gupnet_out_sim_s0/pred_outdoor_test.pkl |
| `D:/advisor_results_1117b/runs/gupnet_out_sim_s1/pred_outdoor_test.pkl` | 2343005 | db1fec64a9afe7ba35582948308c0e716c180797ad1ff2f9b2018fe5f0f98b78 | included at repro/preds/D/advisor_results_1117b/runs/gupnet_out_sim_s1/pred_outdoor_test.pkl |
| `D:/advisor_results_1117b/runs/gupnet_out_sim_s2/pred_outdoor_test.pkl` | 2343005 | 5ef8ac9da681ecdd372e76d03f9617ec3674e50e1cfcd3206af0763bb7ee34bd | included at repro/preds/D/advisor_results_1117b/runs/gupnet_out_sim_s2/pred_outdoor_test.pkl |
| `D:/advisor_results_1117b/runs/gupnet_out_st_anti_s0/pred_outdoor_test.pkl` | 2343005 | 5b2d7612aab82906f308411163499f197db779c0b0484c12b383fbc38fa54b27 | included at repro/preds/D/advisor_results_1117b/runs/gupnet_out_st_anti_s0/pred_outdoor_test.pkl |
| `D:/advisor_results_1117b/runs/gupnet_out_st_anti_s1/pred_outdoor_test.pkl` | 2343005 | 31e896e13695ba4e0cf010a5f78e52c851259177037e0434e9231240611d01eb | included at repro/preds/D/advisor_results_1117b/runs/gupnet_out_st_anti_s1/pred_outdoor_test.pkl |
| `D:/advisor_results_1117b/runs/gupnet_out_st_anti_s2/pred_outdoor_test.pkl` | 2343005 | 07d612998493e9ffd94421d60972ee430fa9ee7565b57ddf4ba0dd808a45b029 | included at repro/preds/D/advisor_results_1117b/runs/gupnet_out_st_anti_s2/pred_outdoor_test.pkl |
| `D:/advisor_results_1117b/runs/monocd_in_kdc_mav_s0/pred_real_test.pkl` | 4366852 | d9017588ddb7962848ce5bfe0d7a30d1735ef8af563773d771e48f93f06dbd4d | included at repro/preds/D/advisor_results_1117b/runs/monocd_in_kdc_mav_s0/pred_real_test.pkl |
| `D:/advisor_results_1117b/runs/monocd_in_kdc_mav_s1/pred_real_test.pkl` | 4366852 | 9e83a0b4cc4a0c7e2abed6ae68dd8cf5016842acf7cda83ea9581a8a0803692b | included at repro/preds/D/advisor_results_1117b/runs/monocd_in_kdc_mav_s1/pred_real_test.pkl |
| `D:/advisor_results_1117b/runs/monocd_in_kdc_mav_s2/pred_real_test.pkl` | 4366852 | 25504c9af698f9a6a9cb78b0f19c03abd89ed7974501affd6b5ef65e9b1a9e74 | included at repro/preds/D/advisor_results_1117b/runs/monocd_in_kdc_mav_s2/pred_real_test.pkl |
| `D:/advisor_results_1117b/runs/monocd_in_sim_s1/pred_real_test.pkl` | 4366852 | f92ee1ba69a54878d12e4275f3e6e8820a50568648cdd3b8a5084bddc0f38cbf | included at repro/preds/D/advisor_results_1117b/runs/monocd_in_sim_s1/pred_real_test.pkl |
| `D:/advisor_results_1117b/runs/monocd_in_sim_s2/pred_real_test.pkl` | 4366852 | f15111946f97f803138df889185bd169087324ce9c04b959ce5afc92225703a2 | included at repro/preds/D/advisor_results_1117b/runs/monocd_in_sim_s2/pred_real_test.pkl |
| `D:/advisor_results_1117b/runs/monocd_in_st_mav_s1/pred_real_test.pkl` | 4366852 | 97704ac9187d555274b64455df30462df5c6f4b2fa8bc21d913ef8af8542454d | included at repro/preds/D/advisor_results_1117b/runs/monocd_in_st_mav_s1/pred_real_test.pkl |
| `D:/advisor_results_1117b/runs/monocd_out_kd_anti_s0/pred_outdoor_test.pkl` | 2343005 | 643623ab9ecc6000b22adc502658c50af6b4733135aaa8465c3edac3cbd110ee | included at repro/preds/D/advisor_results_1117b/runs/monocd_out_kd_anti_s0/pred_outdoor_test.pkl |
| `D:/advisor_results_1117b/runs/monocd_out_kd_anti_s1/pred_outdoor_test.pkl` | 2343005 | 42db605308fbc68c59b3814d90f7b54ef421db17d12eaba11976c46eea4e6b55 | included at repro/preds/D/advisor_results_1117b/runs/monocd_out_kd_anti_s1/pred_outdoor_test.pkl |
| `D:/advisor_results_1117b/runs/monocd_out_kd_anti_s2/pred_outdoor_test.pkl` | 2343005 | 4ccab4a5b8cffa236a304fc8b19d2666e73b5895dcb866ffea4d3aad36b41b30 | included at repro/preds/D/advisor_results_1117b/runs/monocd_out_kd_anti_s2/pred_outdoor_test.pkl |
| `D:/advisor_results_1117b/runs/monocd_out_sim_s0/pred_outdoor_test.pkl` | 2343005 | e39ded9cb5fefbf750a9ae09c63db2b323eba4a1f09c6ff54fae968bdab15440 | included at repro/preds/D/advisor_results_1117b/runs/monocd_out_sim_s0/pred_outdoor_test.pkl |
| `D:/advisor_results_1117b/runs/monocd_out_sim_s1/pred_outdoor_test.pkl` | 2343005 | b992c34d6e1999f4e3639896baf943bfd8ad28770b95d8f4635d2f2a837d4e31 | included at repro/preds/D/advisor_results_1117b/runs/monocd_out_sim_s1/pred_outdoor_test.pkl |
| `D:/advisor_results_1117b/runs/monocd_out_sim_s2/pred_outdoor_test.pkl` | 2343005 | d6b1122f6c9964ee6b09fac8c3c9f8494aa8737ec7c077c88a2bb429f21700eb | included at repro/preds/D/advisor_results_1117b/runs/monocd_out_sim_s2/pred_outdoor_test.pkl |
| `D:/advisor_results_1117b/runs/monocd_out_st_anti_s0/pred_outdoor_test.pkl` | 2343005 | a2b5febce7f2afced614fcab9f6af92b7b3dca0962725122cb1f91a67b317ae8 | included at repro/preds/D/advisor_results_1117b/runs/monocd_out_st_anti_s0/pred_outdoor_test.pkl |
| `D:/advisor_results_1117b/runs/monocd_out_st_anti_s1/pred_outdoor_test.pkl` | 2343005 | a03ba96dc3c109449d31bce62a602ec1efe1b8bd3a8c115dd27236f650087c12 | included at repro/preds/D/advisor_results_1117b/runs/monocd_out_st_anti_s1/pred_outdoor_test.pkl |
| `D:/advisor_results_1117b/runs/monocd_out_st_anti_s2/pred_outdoor_test.pkl` | 2343005 | bed48738386c8a29c84af15299573a27cc79e1391006f5b896de08bc94934553 | included at repro/preds/D/advisor_results_1117b/runs/monocd_out_st_anti_s2/pred_outdoor_test.pkl |
| `D:/advisor_results_1117b/runs/monocop_in_kdc_mav_s0/pred_real_test.pkl` | 2446591 | 689491a9e0e1346a48ba3ccb22f4201daa93cebc580c3338c06c148a9a7f974e | included at repro/preds/D/advisor_results_1117b/runs/monocop_in_kdc_mav_s0/pred_real_test.pkl |
| `D:/advisor_results_1117b/runs/monocop_in_sim_s0/pred_real_test.pkl` | 2446591 | d5553885b301932f09ec8cd2d8ab0a3915dba0945266f2ae160a8e25ab6c1776 | included at repro/preds/D/advisor_results_1117b/runs/monocop_in_sim_s0/pred_real_test.pkl |
| `D:/advisor_results_1117b/runs/monocop_in_st_mav_s0/pred_real_test.pkl` | 2446591 | 81b7945d1d2e482e8475dbb63c3b9b4dec39ba80981d562992949b600f360d2a | included at repro/preds/D/advisor_results_1117b/runs/monocop_in_st_mav_s0/pred_real_test.pkl |
| `D:/advisor_results_1117b/runs/monocop_out_kd_anti_s0/pred_outdoor_test.pkl` | 1293261 | c7ab172b872c060fbb07aa27c7536ab7ae28f0af2d88e76e4cc73ea0805ca11f | included at repro/preds/D/advisor_results_1117b/runs/monocop_out_kd_anti_s0/pred_outdoor_test.pkl |
| `D:/advisor_results_1117b/runs/monocop_out_kd_anti_s1/pred_outdoor_test.pkl` | 1293261 | d0c185ec2bcafb4bc6e774483209ab231cfd7d91632be96f6386366cad6a5657 | included at repro/preds/D/advisor_results_1117b/runs/monocop_out_kd_anti_s1/pred_outdoor_test.pkl |
| `D:/advisor_results_1117b/runs/monocop_out_kd_anti_s2/pred_outdoor_test.pkl` | 1293261 | 2bba169616999a9bfc7fa9643783d10774f980531ec6fad6d797a55f46e17e81 | included at repro/preds/D/advisor_results_1117b/runs/monocop_out_kd_anti_s2/pred_outdoor_test.pkl |
| `D:/advisor_results_1117b/runs/monocop_out_sim_s0/pred_outdoor_test.pkl` | 1293261 | 786e6bf22c3789cb062c50c4e5d93629b4b6d3a39968764c406f14651c359a06 | included at repro/preds/D/advisor_results_1117b/runs/monocop_out_sim_s0/pred_outdoor_test.pkl |
| `D:/advisor_results_1117b/runs/monocop_out_st_anti_s0/pred_outdoor_test.pkl` | 1293261 | 2beaee7efb2d48ae0bda39280a319959e4343ef8c816877ef99d9a3999384792 | included at repro/preds/D/advisor_results_1117b/runs/monocop_out_st_anti_s0/pred_outdoor_test.pkl |
| `D:/advisor_results_1117b/runs/monodetr_in_kdc_mav_s0/pred_real_test.pkl` | 2446591 | 68a468c8162f64a7eb1c44eb0120ceb664fb506bcaa5319c93e39a2b15a97c4c | included at repro/preds/D/advisor_results_1117b/runs/monodetr_in_kdc_mav_s0/pred_real_test.pkl |
| `D:/advisor_results_1117b/runs/monodetr_in_kdc_mav_s1/pred_real_test.pkl` | 2446591 | 3d5c023b9e61cf1a5fe5412bfc5d3f22ce6110c4f82f0b45205063a85357735e | included at repro/preds/D/advisor_results_1117b/runs/monodetr_in_kdc_mav_s1/pred_real_test.pkl |
| `D:/advisor_results_1117b/runs/monodetr_in_kdc_mav_s2/pred_real_test.pkl` | 2446591 | 37d255ed0ea7b399ef81fc7bd44af9a42cbe5cb400d131661873c87b69f74ce1 | included at repro/preds/D/advisor_results_1117b/runs/monodetr_in_kdc_mav_s2/pred_real_test.pkl |
| `D:/advisor_results_1117b/runs/monodetr_in_sim_s0/pred_real_test.pkl` | 2446591 | 7263978901d636ef7b7de7bf993d188105207ce5d723f87d2fbc6007f6ef0d4c | included at repro/preds/D/advisor_results_1117b/runs/monodetr_in_sim_s0/pred_real_test.pkl |
| `D:/advisor_results_1117b/runs/monodetr_in_st_mav_s0/pred_real_test.pkl` | 2446591 | d3b550a004299f6ff6c91cc17fd7239ab7f5fbbe2673d538fac346408f1cf9ea | included at repro/preds/D/advisor_results_1117b/runs/monodetr_in_st_mav_s0/pred_real_test.pkl |
| `D:/advisor_results_1117b/runs/monodetr_out_kd_anti_s0/pred_outdoor_test.pkl` | 1293261 | 7123f59f1ea29b41d2e4a895bbe4b1162c4d35c1c6ee321fb15f24d96f76a840 | included at repro/preds/D/advisor_results_1117b/runs/monodetr_out_kd_anti_s0/pred_outdoor_test.pkl |
| `D:/advisor_results_1117b/runs/monodetr_out_kd_anti_s1/pred_outdoor_test.pkl` | 1293261 | fc86dfbdbd42c8f065943702302c6b9a5135e5aea019555ed757a1fb6a5d1c7b | included at repro/preds/D/advisor_results_1117b/runs/monodetr_out_kd_anti_s1/pred_outdoor_test.pkl |
| `D:/advisor_results_1117b/runs/monodetr_out_kd_anti_s2/pred_outdoor_test.pkl` | 1293261 | 169fa4f5d71321c0e8a93dfe715fed46fa8b9e7a3c503bac0b05066e9e2f1d82 | included at repro/preds/D/advisor_results_1117b/runs/monodetr_out_kd_anti_s2/pred_outdoor_test.pkl |
| `D:/advisor_results_1117b/runs/monodetr_out_sim_s0/pred_outdoor_test.pkl` | 1293261 | c8fe382285d86d9d261ead1971f9275e8a33c7390c0621525749e3a29ecc65f9 | included at repro/preds/D/advisor_results_1117b/runs/monodetr_out_sim_s0/pred_outdoor_test.pkl |
| `D:/advisor_results_1117b/runs/monodetr_out_st_anti_s0/pred_outdoor_test.pkl` | 1293261 | cf21f4040114a0cb50d4c39d0ba74bdebd4966043bcc8f767c05bd0907d5d404 | included at repro/preds/D/advisor_results_1117b/runs/monodetr_out_st_anti_s0/pred_outdoor_test.pkl |
| `D:/advisor_results_1117b/runs/monodgp_in_kdc_mav_s0/pred_real_test.pkl` | 2446591 | b2fb1848ecf73b456992d5a7f5dd6bc21ab90d889d7973e9f191d0e2e2f7e861 | included at repro/preds/D/advisor_results_1117b/runs/monodgp_in_kdc_mav_s0/pred_real_test.pkl |
| `D:/advisor_results_1117b/runs/monodgp_in_kdc_mav_s1/pred_real_test.pkl` | 2446591 | 5d9ee0c1e4754a50d423e3b692d34eea5b1f05a2e64e422f0dae95549a72946b | included at repro/preds/D/advisor_results_1117b/runs/monodgp_in_kdc_mav_s1/pred_real_test.pkl |
| `D:/advisor_results_1117b/runs/monodgp_in_kdc_mav_s2/pred_real_test.pkl` | 2446591 | 66cafb292cd05ff128f0167211ad7250e9c7cd478bf2f48d50a9d22d3b4da062 | included at repro/preds/D/advisor_results_1117b/runs/monodgp_in_kdc_mav_s2/pred_real_test.pkl |
| `D:/advisor_results_1117b/runs/monodgp_in_st_mav_s0/pred_real_test.pkl` | 2446591 | 4f3c6582895b41ab81b7b9d4c92a637bc32116463ea38e1117d709bcc0764490 | included at repro/preds/D/advisor_results_1117b/runs/monodgp_in_st_mav_s0/pred_real_test.pkl |
| `D:/advisor_results_1117b/runs/monodgp_out_kd_anti_s0/pred_outdoor_test.pkl` | 1293261 | 77576d22e77462644353ece7a7f3ab4b489a4fcc3f1aa7a0ce5a7be083fcad5e | included at repro/preds/D/advisor_results_1117b/runs/monodgp_out_kd_anti_s0/pred_outdoor_test.pkl |
| `D:/advisor_results_1117b/runs/monodgp_out_kd_anti_s1/pred_outdoor_test.pkl` | 1293261 | 26298506c0fbb9ee36ada6891ee388aa63cd37dcc34882d409055b1740312b16 | included at repro/preds/D/advisor_results_1117b/runs/monodgp_out_kd_anti_s1/pred_outdoor_test.pkl |
| `D:/advisor_results_1117b/runs/monodgp_out_kd_anti_s2/pred_outdoor_test.pkl` | 1293261 | ba1ebf49780d443b902cb995e9c1ba544515b52205626219cc1f1d7dd26adc49 | included at repro/preds/D/advisor_results_1117b/runs/monodgp_out_kd_anti_s2/pred_outdoor_test.pkl |
| `D:/advisor_results_1117b/runs/monodgp_out_sim_s0/pred_outdoor_test.pkl` | 1293261 | 9609e16bbac82078ecd2bc1eaab3b82bb8c114da545e87b42474f0dc018c7dd4 | included at repro/preds/D/advisor_results_1117b/runs/monodgp_out_sim_s0/pred_outdoor_test.pkl |
| `D:/advisor_results_1117b/runs/monodgp_out_sim_s1/pred_outdoor_test.pkl` | 1293261 | bf1364134a2d6e2f1f4b93c2a6a151f7b29b0cb712062ebf0f5eb87cd2b377b7 | included at repro/preds/D/advisor_results_1117b/runs/monodgp_out_sim_s1/pred_outdoor_test.pkl |
| `D:/advisor_results_1117b/runs/monodgp_out_sim_s2/pred_outdoor_test.pkl` | 1293261 | 3072ac38a4a438f07a81d768722f4698bd2bd72a0c828a412475f2f04caa51b4 | included at repro/preds/D/advisor_results_1117b/runs/monodgp_out_sim_s2/pred_outdoor_test.pkl |
| `D:/advisor_results_1117b/runs/monodgp_out_st_anti_s0/pred_outdoor_test.pkl` | 1293261 | 6349b4b7504a536f7af33847efc5d570962aaecfc3e9c33acad17193236fcce5 | included at repro/preds/D/advisor_results_1117b/runs/monodgp_out_st_anti_s0/pred_outdoor_test.pkl |
| `D:/advisor_results_1117b/runs/monodgp_out_st_anti_s1/pred_outdoor_test.pkl` | 1293261 | 22831423437f4074273052e50e2b77d53e20e436a84f33811783a3dabc5fa6a4 | included at repro/preds/D/advisor_results_1117b/runs/monodgp_out_st_anti_s1/pred_outdoor_test.pkl |
| `D:/advisor_results_1117b/runs/monodgp_out_st_anti_s2/pred_outdoor_test.pkl` | 1293261 | b2fe77762308d17b81cff3d5334e2e14aae02b1dafc7485383d0a155adf4ab38 | included at repro/preds/D/advisor_results_1117b/runs/monodgp_out_st_anti_s2/pred_outdoor_test.pkl |
| `D:/cmpfull_runs/gupnet_s0/pred_real_test.pkl` | 4366852 | 041c865da2f83fa80635a4a4bdc10d32c0e4796afe82302817f34201e57e3a3e | included at repro/preds/D/cmpfull_runs/gupnet_s0/pred_real_test.pkl |
| `D:/cmpfull_runs/gupnet_uda_dann_s0/pred_real_test.pkl` | 4366852 | fc737b8115414e93b16da69f88dfbe5120ce9b8265ea0fdc452397bea9443251 | included at repro/preds/D/cmpfull_runs/gupnet_uda_dann_s0/pred_real_test.pkl |
| `D:/cmpfull_runs/monocd_s0/pred_real_test.pkl` | 4366852 | 94a487d7f0e3fb2f82218125198a947fad679e094c8782ae1829c818455d05c6 | included at repro/preds/D/cmpfull_runs/monocd_s0/pred_real_test.pkl |
| `D:/cmpfull_runs/monodgp_s0/pred_real_test.pkl` | 2446591 | 4ba2c3a6c8c8664ec3e0a97ab117b1b19d19da7becad186d925547ba0a7dba0c | included at repro/preds/D/cmpfull_runs/monodgp_s0/pred_real_test.pkl |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/clean2/buildup/b_det1_test.pkl` | 895975 | 3edf89e8801a1c699a64fb454f3864442fc6c958244e392fb00bfd1eeb018c9a | included at repro/preds/E/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/clean2/buildup/b_det1_test.pkl |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/clean2/buildup/c_students_test.pkl` | 895975 | 6a16969e0fbbbc5cb8c8066723225cac75adb66baa532f647f94db1db7a0adb0 | included at repro/preds/E/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/clean2/buildup/c_students_test.pkl |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads/H3/mav_s0/preds/h3_fused_test.pkl` | 895975 | 01940d75f3168aa986cb306fa9221afb6fce350a1a1641783c0ceac8b2f1700b | included at repro/preds/E/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads/H3/mav_s0/preds/h3_fused_test.pkl |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/ourdet_baseproto/preds/in2mav_dann_test.pkl` | 895975 | ec1cf030268367e7db8b4f75484db8c2c8e80acd8bb91839ff1d898cde4da25d | included at repro/preds/E/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/ourdet_baseproto/preds/in2mav_dann_test.pkl |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/ourdet_baseproto/preds/in2mav_pst_test.pkl` | 895975 | 711cd0f9135b366eebfd22736ab09b3201b3fa6a361809a3246e25cca4f85fed | included at repro/preds/E/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/ourdet_baseproto/preds/in2mav_pst_test.pkl |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/ourdet_baseproto/preds/in2mav_sim_s0_test.pkl` | 895975 | 0512587208c89cd0346325d88c7e4e5dfb3caa9ca214f64a278e2fb7afd5ddab | included at repro/preds/E/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/ourdet_baseproto/preds/in2mav_sim_s0_test.pkl |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/ourdet_baseproto/preds/in2mav_st_test.pkl` | 895975 | ccfa2d286797f3e7408ae726b4218198963f89250df6e850e52e85493307c6cb | included at repro/preds/E/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/ourdet_baseproto/preds/in2mav_st_test.pkl |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/ourdet_baseproto/preds/in2mav_ub_test.pkl` | 895975 | 1dda03c58ded5990475578c173c0918f0ab156a61dd6af6ae51f95d9ff0c7910 | included at repro/preds/E/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/ourdet_baseproto/preds/in2mav_ub_test.pkl |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/ourdet_baseproto/preds/out2antifs_dann_test.pkl` | 445592 | 2c84a324148390d914aa9f2851223263018bb30acf29f94aa14c90fa1d675390 | included at repro/preds/E/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/ourdet_baseproto/preds/out2antifs_dann_test.pkl |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/ourdet_baseproto/preds/out2antifs_pst_test.pkl` | 445592 | 011183c0181adf119a62ac46c7beba16beb6b1204f57c90e84634414d0aca0ec | included at repro/preds/E/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/ourdet_baseproto/preds/out2antifs_pst_test.pkl |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/ourdet_baseproto/preds/out2antifs_sim_s0_test.pkl` | 445592 | c6afa6e39d137cca808d291d45cabf6f9913fc43cbe5bfd3cc28fbd28b138a12 | included at repro/preds/E/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/ourdet_baseproto/preds/out2antifs_sim_s0_test.pkl |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/ourdet_baseproto/preds/out2antifs_st_test.pkl` | 445592 | 9da464167e27c149fb66a748952a91d9d704656d420e80d03451144f4999b821 | included at repro/preds/E/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/ourdet_baseproto/preds/out2antifs_st_test.pkl |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/ourdet_baseproto/preds/out2antifs_ubzoom_test.pkl` | 445592 | 0a2cb069a54bcd7d512d10b153317545c1408e297fbdf639c408e617cc850cb9 | included at repro/preds/E/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/ourdet_baseproto/preds/out2antifs_ubzoom_test.pkl |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/seeds/antifs_full_s10/preds/pcmd_fused_test.pkl` | 445592 | b667a7eafe6f860e0068528f11e57b07449420d37dd672a57526e1b2729c0418 | included at repro/preds/E/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/seeds/antifs_full_s10/preds/pcmd_fused_test.pkl |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/seeds/antifs_full_s20/preds/pcmd_fused_test.pkl` | 445592 | 27dbbb9c58a3bfcca2006283e1dacb1ca67a53f05708d329675e22aa381cc914 | included at repro/preds/E/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/seeds/antifs_full_s20/preds/pcmd_fused_test.pkl |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/seeds/antifs_none_s10/preds/pcmd_fused_test.pkl` | 445592 | 929a1aa938c71da3a12a0daf9f5a47d643d692dca0cdba1518f53ec7e90f3863 | included at repro/preds/E/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/seeds/antifs_none_s10/preds/pcmd_fused_test.pkl |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/seeds/antifs_none_s20/preds/pcmd_fused_test.pkl` | 445592 | 35db69276dc57bcafb934a253b3fd6ce8b954e0403be1e530dc50c30e5fa00e3 | included at repro/preds/E/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/seeds/antifs_none_s20/preds/pcmd_fused_test.pkl |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/seeds/mav_full_s20/preds/pcmd_fused_test.pkl` | 895975 | cfb676d0a7c64f976ebb94818fcbc8fd60d0f960a08fb650015b764c6e67d71a | included at repro/preds/E/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/seeds/mav_full_s20/preds/pcmd_fused_test.pkl |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/seeds/mav_none_s10/preds/pcmd_fused_test.pkl` | 895975 | 70769055e965c1e3c05b08f02bd5b765a1ab71293d5ff6d3b0ce8e92f9dd8f4b | included at repro/preds/E/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/seeds/mav_none_s10/preds/pcmd_fused_test.pkl |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/seeds/mav_none_s20/preds/pcmd_fused_test.pkl` | 895975 | bfcdf30029cb6af3dc5c9e2d05f80cdc1368bd6d384f3972f74d138e9ca65fc3 | included at repro/preds/E/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/seeds/mav_none_s20/preds/pcmd_fused_test.pkl |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree/E1/antifs_s0/preds/e1_fused_test.pkl` | 445592 | 4b7ce82461625ea21df9b23700ca79bd908573fbef85569d2680dc61e9e08606 | included at repro/preds/E/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree/E1/antifs_s0/preds/e1_fused_test.pkl |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree/E1/antifs_s10/preds/e1_fused_test.pkl` | 445592 | 09334e360b6e37885eb55478f64d50d80d1ffc883bcb40c48a941c42fc980cd6 | included at repro/preds/E/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree/E1/antifs_s10/preds/e1_fused_test.pkl |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree/E1/antifs_s20/preds/e1_fused_test.pkl` | 445592 | 0f1fce8e5d44b1cb3d84ba849d2c9b651a1e4c13e12eea6135698e4b03b750d8 | included at repro/preds/E/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree/E1/antifs_s20/preds/e1_fused_test.pkl |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree/E1/mav_s0/preds/e1_fused_test.pkl` | 895975 | 248fb62257be9673976e7dd46c818a141b33b3d4773c43c59256ba74b3a43689 | included at repro/preds/E/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree/E1/mav_s0/preds/e1_fused_test.pkl |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree/E1/mav_s10/preds/e1_fused_test.pkl` | 895975 | 189edfafc858c46ad40120bdeac7d0ee558d9724915eb1c5b65d0b4f238b9b59 | included at repro/preds/E/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree/E1/mav_s10/preds/e1_fused_test.pkl |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree/E1/mav_s20/preds/e1_fused_test.pkl` | 895975 | f69336ffc0dfd25e3fe9a631c99e40d11d4dcdda140ba7c4678dc3fdc9af800f | included at repro/preds/E/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree/E1/mav_s20/preds/e1_fused_test.pkl |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree/E1/pooled_antifs_vsPST.json` | 5534 | e77660b979146c0e78ac14404b33fa7e88023a8ae3b88ce9f3381522dd93d45c | included at repro/preds/E/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree/E1/pooled_antifs_vsPST.json |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree/E1/pooled_mav_vsPST.json` | 5378 | 58fbc20d203fb4f5091af76fbdbd7d3a739cc13454cc79b3e84ef370d55bc66b | included at repro/preds/E/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree/E1/pooled_mav_vsPST.json |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree/E4/antifs_s0/preds/e4_fused_test.pkl` | 445592 | 8457bfbba2d69d4454f4c7b475c1cb937ce1785f093dcb1be332ed2476f13761 | included at repro/preds/E/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree/E4/antifs_s0/preds/e4_fused_test.pkl |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree/E4/antifs_s10/preds/e4_fused_test.pkl` | 445592 | 50955ed3a44fcf2a12ca9b1e1cd978c3885e28303562f965cdcf918480676b7b | included at repro/preds/E/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree/E4/antifs_s10/preds/e4_fused_test.pkl |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree/E4/antifs_s20/preds/e4_fused_test.pkl` | 445592 | 5cf19bea199892c7bf77b21fe3343ef2de6dab82de68f3e9a3f85298d65f3352 | included at repro/preds/E/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree/E4/antifs_s20/preds/e4_fused_test.pkl |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree/E4/mav_s0/preds/e4_fused_test.pkl` | 895975 | 2ff09a4234aad596bfe61a192f18d1c148abbd8f7099fb76b9f9fd6589e1f24f | included at repro/preds/E/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree/E4/mav_s0/preds/e4_fused_test.pkl |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree/E4/mav_s10/preds/e4_fused_test.pkl` | 895975 | ff904ffab15a462cfa8ce49a17bb8a14e274f95d59a62e7e9154fdd545c0bde0 | included at repro/preds/E/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree/E4/mav_s10/preds/e4_fused_test.pkl |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree/E4/mav_s20/preds/e4_fused_test.pkl` | 895975 | be9d3a2bcde7e20e35044c0bbce103c99ce92e8682b5c6fe8bc548e9bfe2b4a6 | included at repro/preds/E/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree/E4/mav_s20/preds/e4_fused_test.pkl |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sup3/run/preds/pcmd_fused_test.pkl` | 895975 | e07cd8e32ea42a8267d789d8ca73df0a13ba3e536074954801457b8e125f9106 | included at repro/preds/E/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sup3/run/preds/pcmd_fused_test.pkl |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/srcx/in2mav/preds/pcmd_fused_test.pkl` | 895975 | 4574b709a1694e9e77449c145eb356473bf80a5bd495ff0c0160b882f0f7f07b | included at repro/preds/E/Open3DUAVDet/output/camnorm/geo1004/gproto/srcx/in2mav/preds/pcmd_fused_test.pkl |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/srcx/in2mav/preds/srconly_test.pkl` | 895975 | 33061861003e27eee31e431e04eeb8d0483c65cbb5fd89fec2f8cf17096c02a3 | included at repro/preds/E/Open3DUAVDet/output/camnorm/geo1004/gproto/srcx/in2mav/preds/srconly_test.pkl |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/srcx/in2mav_none/preds/pcmd_fused_test.pkl` | 895975 | ab0b8cddc76b873e8119250dbc282f18437094376f6dc701a186bacc107946e4 | included at repro/preds/E/Open3DUAVDet/output/camnorm/geo1004/gproto/srcx/in2mav_none/preds/pcmd_fused_test.pkl |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/srcx/out2antifs/preds/pcmd_fused_test.pkl` | 445592 | 70cac1563d1b03233fdaf42027ce2af4e628c3653375e73ecec9307172a15a79 | included at repro/preds/E/Open3DUAVDet/output/camnorm/geo1004/gproto/srcx/out2antifs/preds/pcmd_fused_test.pkl |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/srcx/out2antifs_k249/preds/pcmd_fused_test.pkl` | 445592 | 2177a800d7dcba2e1a6c2b718960ba9d692ebbe52b2e73844a98421833a3ab17 | included at repro/preds/E/Open3DUAVDet/output/camnorm/geo1004/gproto/srcx/out2antifs_k249/preds/pcmd_fused_test.pkl |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/srcx/out2antifs_k249/preds/srconly_test.pkl` | 445592 | d42238f40c54d89b5f03798889ed53674b6aea8ad4f77bf71d69271faa9a9028 | included at repro/preds/E/Open3DUAVDet/output/camnorm/geo1004/gproto/srcx/out2antifs_k249/preds/srconly_test.pkl |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/srcx/out2antifs_none/preds/pcmd_fused_test.pkl` | 445592 | db95f6d74fd46ce21977a3b805976d0a3f41e3b506c80611752e5e3dbe76ad14 | included at repro/preds/E/Open3DUAVDet/output/camnorm/geo1004/gproto/srcx/out2antifs_none/preds/pcmd_fused_test.pkl |

## Not copied: data, caches, checkpoints, large or binary files found in the copied code folders

Directories marked "dir (not traversed)" hold crops, caches, checkpoints, records or predictions of the runs (pruned by name). Files: size, mtime, sha256 (sha256 only when <= 50 MB).

| path | size | mtime | reason / ext | sha256 |
|---|---|---|---|---|
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/srcx/pipeline_figure_style_1112.webp` | 205510 | 2026-11-12 02:32 | .webp | d8198ab560cbd66abca12803d19150a84a5acf5f27391c8d329a3a9aabb4cfa0 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/srcx/lib/__pycache__/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/srcx/in2mav/logs\run_x.stdout` | 1367 | 2026-11-13 08:24 | .stdout | 8d81a4884e3dadd901779f14445c4ce91b916e2410146980a9d290292b6ef903 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/srcx/in2mav_none/logs\run_x.stdout` | 994 | 2026-11-13 12:06 | .stdout | 9625da8d98d82d652863ad402067e076bc705beb98adf4279098c58fab27d4cc |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/srcx/out2antifs/logs\run_x.stdout` | 984 | 2026-11-12 05:04 | .stdout | e703779c4e909579e24a2f44fbb2e83d778129cf5d80d1e1ab78b3fea9c21a99 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/srcx/out2antifs_none/logs\run_x.stdout` | 1273 | 2026-11-12 13:31 | .stdout | f4f218dddff0b22556995dab77b18629d21ac164c30f223742fffd4424b1f0e8 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/srcx/logs\stage1_det_sim_out_r34_geo_ramp_out_s1.done` | 7 | 2026-11-11 12:17 | .done | 28d3b9e880a77975493dc7e359144c0295a4f694cfe0af4f928c22307bc5c320 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/srcx/logs\stage1_det_sim_out_r34_geo_ramp_out_s2.done` | 7 | 2026-11-11 12:39 | .done | 28d3b9e880a77975493dc7e359144c0295a4f694cfe0af4f928c22307bc5c320 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/B3\both\kps_te1.pkl` | 209355 | 2026-11-14 03:48 | .pkl | 9062b0fa6cd45688fc11cbb441c9d81a0c88471d8a8c1d23ec3798ea4781c59c |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/B3\both\kps_tr1.pkl` | 616223 | 2026-11-14 03:48 | .pkl | 820b0bb2ef3a6bd21940616de219d39d735f8539c20994594de5033b50f359c8 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/B3\both\kps_val1.pkl` | 110298 | 2026-11-14 03:48 | .pkl | da1ce186f10b1123f4b2656f76675db6a2f538dd1c4d4fa5c801ab46cb4f478a |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/B3/both/ou/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/B3/both/preds/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/B3/both/recs/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/B3/both/ref/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/B3\both\work\oc_te1.pkl` | 1707425 | 2026-11-14 03:48 | .pkl | bf3e21e180d2eebc48a3d57a843983e32a64e9f786f7fee3d4f5af8eececb7ed |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/B3\both\work\oc_tr1.pkl` | 4631227 | 2026-11-14 03:48 | .pkl | 1f5fd36e5b2e1a811c5c2c4fa5a734a53f7d22b28065b9f8493439b80a5c32f8 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/B3\both\work\oc_val1.pkl` | 839492 | 2026-11-14 03:48 | .pkl | 4ed7912b9fdc5c8bdc072cc8e63f45d8e72b34ace7acd4276040b868477b4e64 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/B3\both\work\pseudo_tr1.npz` | 5574350 | 2026-11-14 03:48 | .npz | 896e68a62b5d364efaae0ee4c77e396bbac6cfe1513238199695085432410934 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/B3/calib/runs/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/B3\feat\kps_te1.pkl` | 209355 | 2026-11-14 03:48 | .pkl | 9062b0fa6cd45688fc11cbb441c9d81a0c88471d8a8c1d23ec3798ea4781c59c |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/B3\feat\kps_tr1.pkl` | 616223 | 2026-11-14 03:48 | .pkl | 820b0bb2ef3a6bd21940616de219d39d735f8539c20994594de5033b50f359c8 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/B3\feat\kps_val1.pkl` | 110298 | 2026-11-14 03:48 | .pkl | da1ce186f10b1123f4b2656f76675db6a2f538dd1c4d4fa5c801ab46cb4f478a |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/B3/feat/ou/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/B3/feat/preds/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/B3/feat/recs/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/B3/feat/ref/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/B3\feat\work\oc_te1.pkl` | 1707425 | 2026-11-14 03:48 | .pkl | bf3e21e180d2eebc48a3d57a843983e32a64e9f786f7fee3d4f5af8eececb7ed |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/B3\feat\work\oc_tr1.pkl` | 4631227 | 2026-11-14 03:48 | .pkl | 1f5fd36e5b2e1a811c5c2c4fa5a734a53f7d22b28065b9f8493439b80a5c32f8 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/B3\feat\work\oc_val1.pkl` | 839492 | 2026-11-14 03:48 | .pkl | 4ed7912b9fdc5c8bdc072cc8e63f45d8e72b34ace7acd4276040b868477b4e64 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/B3\feat\work\pseudo_tr1.npz` | 5574350 | 2026-11-14 03:48 | .npz | 896e68a62b5d364efaae0ee4c77e396bbac6cfe1513238199695085432410934 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/B3/lib/__pycache__/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/B4/lib/__pycache__/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/B4\run\kps_te1.pkl` | 209355 | 2026-11-14 09:00 | .pkl | a21a572dfc02f475ff07168d6bfde44cb9bc975d17555f29d337656d55607f92 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/B4\run\kps_tr1.pkl` | 616223 | 2026-11-14 08:59 | .pkl | ade56e8e69ef037d4262f12e04876020067eb17b9ca935438c5a86d8e4033601 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/B4\run\kps_val1.pkl` | 110298 | 2026-11-14 08:59 | .pkl | 741322f157f949417f32de0bd7f9ec86bcc81b646ec4f6a4c24255c50ae3aad9 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/B4/run/ou/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/B4/run/preds/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/B4/run/recs/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/B4/run/ref/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/B4\run\wsim_cache.pkl` | 1912979 | 2026-11-14 05:08 | .pkl | 3b27c15c20c112b28d385f9c6b7506d692c5a90850de16d51afc5ec24ada878c |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/B4\run\wsim_read.pkl` | 10352728 | 2026-11-14 10:22 | .pkl | c94afb7008352b502112502f248e26b50136fc86155548f3f0b7eb2131901da7 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/B8\s10\kps_te1.pkl` | 209355 | 2026-11-14 03:58 | .pkl | 5643b8e63d5ee793f382b3231cb0ceedba28c7f1c93f7c1ae45c6c0fbac8fb93 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/B8\s10\kps_tr1.pkl` | 616223 | 2026-11-14 03:58 | .pkl | aa90395c7519b78721e65d5e28119ee2e6d7619300be670aa1cb323ccaf1680e |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/B8\s10\kps_val1.pkl` | 110298 | 2026-11-14 03:58 | .pkl | 56ee82b406127dba198fb20d4732cbf0c202c365b1a3f6a8377329b104d3f37e |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/B8/s10/lib/__pycache__/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/B8/s10/ou/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/B8/s10/preds/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/B8/s10/recs/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/B8/s10/ref/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/B8\s10\work\oc_te1.pkl` | 1707425 | 2026-11-14 03:58 | .pkl | ac54c31b27f9d16aac02f1142b3e46a883e8e176588fcc5d119263f1f7f2430f |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/B8\s10\work\oc_tr1.pkl` | 4631227 | 2026-11-14 03:58 | .pkl | 748fea8d603c77ae4174aa47a64cb1026d0af02e84e31ea617f6b926e0619f0e |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/B8\s10\work\oc_val1.pkl` | 839492 | 2026-11-14 03:58 | .pkl | 09d5da21a98d9b710c62cb525ccf226ea05962c8c692fe149d1f769e33428021 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/B8\s20\kps_te1.pkl` | 209355 | 2026-11-14 06:10 | .pkl | e26b0bd9cfb051fa807bccf680caea54c2c3e14993ed24586facc9639a87024e |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/B8\s20\kps_tr1.pkl` | 616223 | 2026-11-14 06:10 | .pkl | 0345ce60d61d9c0333f549c2c424781ac57c683bf1ccb9fc586557dbfc0d9dea |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/B8\s20\kps_val1.pkl` | 110298 | 2026-11-14 06:10 | .pkl | 7ddb73da9639923326959623c46384c53311075c35bbb55497d41413d5c7de16 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/B8/s20/lib/__pycache__/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/B8/s20/ou/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/B8/s20/preds/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/B8/s20/recs/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/B8/s20/ref/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/B8\s20\work\oc_te1.pkl` | 1707425 | 2026-11-14 06:11 | .pkl | 74c25b66feaf6aa523b055acd46b33bfc7ee5cbc0a9d02027b0c4f28ade512af |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/B8\s20\work\oc_tr1.pkl` | 4631227 | 2026-11-14 06:11 | .pkl | e3183fd06578d5b509280a6e076a2089677c213bf0032c2526c31db985b9a6e9 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/B8\s20\work\oc_val1.pkl` | 839492 | 2026-11-14 06:11 | .pkl | b3adb3318b740f196329b647ad76ee026cf136e733d2b109fc6668e41270dda0 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/M12\M1\repl\nl075_test_kzt7q\kz_members_nl075_test_kzt7q.npz` | 44050908 | 2026-11-13 04:43 | .npz | 723e92ffc4e0bc3926fa0f44ead0527eab741e6bc341feb9b32d3424de82b1cb |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/M12\M1\repl\nl075_test_kzt7q\real_ens5flip_nl075_test_kzt7q.pkl` | 2037197 | 2026-11-13 04:43 | .pkl | dfd4211744a5cf18b246999588e3bd5e531770e628e2b25ee329a791eb705e35 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/M12\M1\repl\nl075_test_kzt7q\sealed_gt_nl075_test_kzt7q.pkl` | 727721 | 2026-11-13 04:43 | .pkl | 877399fa0b7adf17193f5b4df614ccf4dc9a7d5c3eea47e458b7fc59361b2186 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/M12\M1\repl\nl075_val_kzt7q\kz_members_nl075_val_kzt7q.npz` | 21441244 | 2026-11-13 04:51 | .npz | 36069cfc423e93957435ba58c1d9135e7263054ab995bc52af8903b7c165b6ac |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/M12\M1\repl\nl075_val_kzt7q\real_ens5flip_nl075_val_kzt7q.pkl` | 1000310 | 2026-11-13 04:51 | .pkl | bb5bfec4e50dda6e3d30aefd2849e33785e2f9f97fe157ed973c3d9fe640c640 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/M12\M1\repl\nl075_val_kzt7q\sealed_gt_nl075_val_kzt7q.pkl` | 362425 | 2026-11-13 04:51 | .pkl | 95789072a416cc6418a9ccd2b4b9f9389bb38eda44bb482b4ea32572cf2cf399 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/M12\M1\repro_full\fe_deploy_fused_test_P.pkl` | 422800 | 2026-11-13 04:08 | .pkl | 12eb1577698861f469063b504dd8fa3211683b5fafe0eaf58733f559814cd385 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/M12\M1\repro_full\fe_deploy_fused_val_P.pkl` | 205959 | 2026-11-13 04:08 | .pkl | fc333e13017e7f93c7920ce9a55a15fe398e784bd402e908f60776bdf40d11e5 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/M12\M1\repro_full\fe_no_fusion_w0_test_P.pkl` | 422800 | 2026-11-13 04:08 | .pkl | 45023ed1516b7a057a932bf2f688ff3cbd2e23ead3a27b586b7e988f7be5a4d2 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/M12\M1\repro_full\fe_no_fusion_w0_val_P.pkl` | 205959 | 2026-11-13 04:08 | .pkl | 92315d8d4b47fc476f0480985f36e2a772e5c2ad0abdb81ab0df4d55839cc40c |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/M12\M1\single\kps\shape_test_nl075_k0.pkl` | 209355 | 2026-11-13 05:06 | .pkl | 09aeb2cd32fbb356fc2a444a3244b305c0e329e7a813286a8df273569403595e |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/M12\M1\single\kps\shape_train_nl075_k0.pkl` | 68501 | 2026-11-13 05:06 | .pkl | e16cad82c38aeb7035330a6b4d347a994446fa98e7c0b0581a9a1ad0b0823650 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/M12\M1\single\kps\shape_val_nl075_k0.pkl` | 110298 | 2026-11-13 05:03 | .pkl | 4107b211db7b3d832c9d0d04905d7345abc9b8da70565728bb7adfdb3a9a3d85 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/M12\M1\single\tags\nl075_test_ouS1\kz_members_nl075_test_ouS1.npz` | 19570908 | 2026-11-13 04:56 | .npz | 3077f6b857cb4c8eb725373e5a64c778e08c876ab17174ba1fe0cca89999b550 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/M12\M1\single\tags\nl075_val_ouS1\kz_members_nl075_val_ouS1.npz` | 9527644 | 2026-11-13 04:57 | .npz | 325953b4bc4d514ecbbefeca510db5711a036ef4e22c5cdc74e0d1c6646ec984 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/M12\M1\single\unc\unc_test_z2.pkl` | 1404414 | 2026-11-13 04:58 | .pkl | cc46339f3cdd550c1e79d07a4f1f43f0ff8822ef3582bbc2b780b0b2e7781c72 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/M12\M1\single\unc\unc_train_z2.pkl` | 3819812 | 2026-11-13 05:02 | .pkl | 486f6fb9883ca81a307e1073b8e4557414e0b6b6acde15ca2c63a24ad1be197b |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/M12\M1\single\unc\unc_val_z2.pkl` | 691623 | 2026-11-13 04:58 | .pkl | d3e574ead800fe2fb4484bac34e47ae508e85f8680f1da0b0af97a2257f7bf4c |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/M12\M1\single_ens\ens_test.pkl` | 1404934 | 2026-11-13 04:15 | .pkl | b162aea8894f877463579f6a243a49dfe548bc268520660c15c4e86b76d18350 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/M12\M1\single_ens\ens_val.pkl` | 692260 | 2026-11-13 04:16 | .pkl | b3d23af5c673809dc0487c9c5eae5c46cd05381bf94f2a5eacce7c8f421f56e3 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/M12\M1\single_result\fe_deploy_fused_test_P.pkl` | 422800 | 2026-11-13 05:06 | .pkl | 30c87f9721b3b9b2965bb008bf7f341d5d54dff8b36176030b71eab9e40972e0 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/M12\M1\single_result\fe_deploy_fused_val_P.pkl` | 205959 | 2026-11-13 05:06 | .pkl | 074fa9f0433bca7e8eab9e559e5e0f83a9a52b0b542dd8c292ad2f50bfbe5e3d |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/M12\M1\single_test.pkl` | 1404934 | 2026-11-13 04:12 | .pkl | b162aea8894f877463579f6a243a49dfe548bc268520660c15c4e86b76d18350 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/M12\M1\single_train.pkl` | 3820515 | 2026-11-13 04:29 | .pkl | 0977e217059de065f664c504088ad10aa0b3ad46f042b8320e1c84a51aa3c62f |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/M12\M1\single_val.pkl` | 692260 | 2026-11-13 04:16 | .pkl | b3d23af5c673809dc0487c9c5eae5c46cd05381bf94f2a5eacce7c8f421f56e3 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/M12\M2\sim3nb_test.pkl` | 1404934 | 2026-11-13 04:12 | .pkl | 8c3d3654f2e72d997c8afa38c58dc80eb999fe234f625c5fd2e284ace9f5f11c |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/M12\anti\repro\kps_te1.pkl` | 70185 | 2026-11-13 04:10 | .pkl | 04a1d3b583f44dd42ac055dfd18ea4e239858ed0ce70b541ef01112b65c9be52 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/M12\anti\repro\kps_tr1.pkl` | 95245 | 2026-11-13 04:10 | .pkl | 6a658b3667380b92ede2c8ad3aea896ebd5483bb158b7a59f54faa319281b74a |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/M12\anti\repro\kps_val1.pkl` | 30842 | 2026-11-13 04:10 | .pkl | a68623ada3cc328e6633e08556c31176ee350d50bcedf173f5788a1fea8cee09 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/M12/anti/repro/ou/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/M12/anti/repro/preds/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/M12/anti/repro/recs/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/M12/anti/repro/ref/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/M12\anti\repro\work\oc_te1.pkl` | 889244 | 2026-11-13 04:10 | .pkl | d0296372ced54cde1290dfa516f815d4248b696131e30f12294f49a6ed775070 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/M12\anti\repro\work\oc_val1.pkl` | 389326 | 2026-11-13 04:10 | .pkl | 67f9424304a1cde02a83185ece9795ed26ada1d6aec82971cbe552f072702871 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/M12\anti\single\kps_te1.pkl` | 70185 | 2026-11-13 04:34 | .pkl | 7eef22fa3502fb7a8d34d21788f34cb27fbde18423220d86406db93b9c00406a |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/M12\anti\single\kps_tr1.pkl` | 95245 | 2026-11-13 04:33 | .pkl | e43ce9f257eb35c485940f9db1089daf909380d8672f8a6e0bca118203228909 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/M12\anti\single\kps_val1.pkl` | 30842 | 2026-11-13 04:33 | .pkl | 8f6679c95679906a0da9f01692c0d39408894b6911643e027943ae70fda5f66a |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/M12/anti/single/ou/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/M12/anti/single/preds/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/M12/anti/single/recs/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/M12/anti/single/ref/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/M12\anti\single\work\oc_te1.npy` | 1160773760 | 2026-11-13 04:31 | .npy |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/M12\anti\single\work\oc_te1.pkl` | 889244 | 2026-11-13 04:31 | .pkl | 8d1479da577d54628b187311e2eb1d917df34064d4d0e3f9fa993e312748ee97 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/M12\anti\single\work\oc_val1.npy` | 507838592 | 2026-11-13 04:25 | .npy |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/M12\anti\single\work\oc_val1.pkl` | 389326 | 2026-11-13 04:25 | .pkl | 8d5e84675972c2e77bce377abdddef2ea0439f6b2e644931818d88567f73aac6 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/N1\real_val_preds_st2nb.pkl` | 1238029 | 2026-11-13 02:52 | .pkl | 54517973d228dd3d712ea0f6673a594ef81423bb7281ccf3ee92578b6edbacc4 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/N1\val_deltas.pkl` | 968365 | 2026-11-13 02:55 | .pkl | 36ea9a04d4ee2bb8d732ba767b2b10d713d3b47b2be290ea0022ea9e291f4b1c |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/N1clean\chk_s5_train300.pkl` | 24543 | 2026-11-13 14:30 | .pkl | 4612230f7690891ce230a161b2354568714da9f8f5ecce5fb440ca071ff4dec0 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/N1clean\chk_s6_train300.pkl` | 24543 | 2026-11-13 14:37 | .pkl | 57a76df2f1de68ba7a9d5889acd323e0e7dbbb7aac8d0a346d8afc7ad376764e |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/N1clean\st2_s5_det0_val.pkl` | 175724 | 2026-11-13 14:37 | .pkl | 374fe21322e1aaaddb55fcbb7c067dc4f2e6b8da54756927a114758f64d84f46 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/N1clean\st2_s6_det0_val.pkl` | 175724 | 2026-11-13 14:38 | .pkl | 0712dd4069ab798937f6d95865d875c77b162a9c1e5f61174398fef15406d940 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/boxhull/__pycache__/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/boxhull\in2mav\kps_te1.pkl` | 209355 | 2026-11-14 02:50 | .pkl | 9062b0fa6cd45688fc11cbb441c9d81a0c88471d8a8c1d23ec3798ea4781c59c |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/boxhull\in2mav\kps_tr1.pkl` | 616223 | 2026-11-14 02:50 | .pkl | 820b0bb2ef3a6bd21940616de219d39d735f8539c20994594de5033b50f359c8 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/boxhull\in2mav\kps_val1.pkl` | 110298 | 2026-11-14 02:50 | .pkl | da1ce186f10b1123f4b2656f76675db6a2f538dd1c4d4fa5c801ab46cb4f478a |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/boxhull/in2mav/ou/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/boxhull/in2mav/preds/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/boxhull/in2mav/recs/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/boxhull/in2mav/ref/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/boxhull\in2mav\work\oc_te1.pkl` | 1707425 | 2026-11-14 02:50 | .pkl | bf3e21e180d2eebc48a3d57a843983e32a64e9f786f7fee3d4f5af8eececb7ed |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/boxhull\in2mav\work\oc_tr1.pkl` | 4631227 | 2026-11-14 02:50 | .pkl | 1f5fd36e5b2e1a811c5c2c4fa5a734a53f7d22b28065b9f8493439b80a5c32f8 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/boxhull\in2mav\work\oc_val1.pkl` | 839492 | 2026-11-14 02:50 | .pkl | 4ed7912b9fdc5c8bdc072cc8e63f45d8e72b34ace7acd4276040b868477b4e64 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/boxhull/lib/__pycache__/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/boxhull\out2antifs\kps_te1.pkl` | 70185 | 2026-11-14 06:43 | .pkl | 04a1d3b583f44dd42ac055dfd18ea4e239858ed0ce70b541ef01112b65c9be52 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/boxhull\out2antifs\kps_tr1.pkl` | 95245 | 2026-11-14 06:43 | .pkl | 6a658b3667380b92ede2c8ad3aea896ebd5483bb158b7a59f54faa319281b74a |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/boxhull\out2antifs\kps_val1.pkl` | 30842 | 2026-11-14 06:43 | .pkl | a68623ada3cc328e6633e08556c31176ee350d50bcedf173f5788a1fea8cee09 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/boxhull/out2antifs/ou/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/boxhull/out2antifs/preds/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/boxhull/out2antifs/recs/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/boxhull/out2antifs/ref/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/boxhull\out2antifs\work\oc_te1.pkl` | 889244 | 2026-11-14 06:43 | .pkl | d0296372ced54cde1290dfa516f815d4248b696131e30f12294f49a6ed775070 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/boxhull\out2antifs\work\oc_tr1.pkl` | 1123068 | 2026-11-14 06:43 | .pkl | 746685b1825e702e8e4f1a26d0ae22dd32a85f7f3f3d23e95aa60f3b1e493151 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/boxhull\out2antifs\work\oc_val1.pkl` | 389326 | 2026-11-14 06:43 | .pkl | 67f9424304a1cde02a83185ece9795ed26ada1d6aec82971cbe552f072702871 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/breakdown\meta\antifs_test_M400.npz` | 13498 | 2026-11-13 15:42 | .npz | 54f187a34f4988e8eba4d010d8e6be2dee4c86fead2eebb9a025bb8fbe75ddb8 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/breakdown\meta\antifs_test_M4E.npz` | 660666 | 2026-11-13 15:42 | .npz | f4e5c25dc44e5867a9dbd945c8d513ca7144417456b4962a691c5cd078c64fd6 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/breakdown\meta\mav_test_mavic2.npz` | 502688 | 2026-11-13 15:42 | .npz | deeacf62672969eb2923ebc98ae3b7f5f8b712a863402b492eec6e56300a0cb9 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/breakdown\meta\mav_test_phantom4.npz` | 1364048 | 2026-11-13 15:42 | .npz | 6fc0ea32d204653b05cf8828eeea4eb59a103fa83b58437aa26028168b05bb0b |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/breakdown\meta\mav_val_mavic2.npz` | 490660 | 2026-11-14 12:01 | .npz | a55713d936f6033703e5bc0b894568a813ac0db05e0ab906628eceb6a494e568 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/breakdown\meta\mav_val_phantom4.npz` | 420044 | 2026-11-14 12:01 | .npz | a8fae793c4eaa75077411954c67b758f4443d42c522b2fab82c54632249dedb7 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/clean\single\kps_te1.pkl` | 209355 | 2026-11-13 09:44 | .pkl | 6ffa3ff5dd46c0bcdb83c13978107365a57812b8cd411620cf3d5e9660d69aac |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/clean\single\kps_tr1.pkl` | 616223 | 2026-11-13 09:44 | .pkl | 89e25c0dbca63f574b45f1efe8ee873f3edad11849b369d49717627385e99981 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/clean\single\kps_val1.pkl` | 110298 | 2026-11-13 09:44 | .pkl | 72efe40e255669073f056a42af6f43b53473c810b3b9bfcc6b21c163821c29ce |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/clean\single\logs\run_single.stdout` | 983 | 2026-11-13 09:53 | .stdout | 72286db08c55f0edf72a9be2bcf436b1d61785c3a76319c273fc2270afc9cd9b |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/clean/single/ou/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/clean/single/preds/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/clean/single/recs/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/clean/single/ref/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/clean\single\work\oc_te1.npy` | 2123366528 | 2026-11-13 09:37 | .npy |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/clean\single\work\oc_te1.pkl` | 1707425 | 2026-11-13 09:37 | .pkl | 242858369aadf1db4f869bbb936a7c7ec1eb07a7e6e78c5b23bb847e593b5293 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/clean\single\work\oc_val1.npy` | 1033371776 | 2026-11-13 09:19 | .npy |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/clean\single\work\oc_val1.pkl` | 839492 | 2026-11-13 09:19 | .pkl | f5f02c8b955246e878e850a0e36f49516a42a1bc333601df5a2a5b8ae6de0195 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/clean2\buildup\b_det1_test.pkl` | 895975 | 2026-11-13 10:54 | .pkl | 3edf89e8801a1c699a64fb454f3864442fc6c958244e392fb00bfd1eeb018c9a |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/clean2\buildup\b_det1_val.pkl` | 444521 | 2026-11-13 10:54 | .pkl | e94ffcf9a4cff40c626a426d9998efdc3d663590e4e5a3227af6aefbee05eea7 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/clean2\buildup\c_students_test.pkl` | 895975 | 2026-11-13 10:54 | .pkl | 6a16969e0fbbbc5cb8c8066723225cac75adb66baa532f647f94db1db7a0adb0 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/clean2\buildup\c_students_val.pkl` | 444521 | 2026-11-13 10:54 | .pkl | 1968853a5b0e52a1eedd8bd905be20ce601f177bbf151ba56aee17621c51bba0 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/clean2\buildup\cp_students_rk_test.pkl` | 895975 | 2026-11-13 10:54 | .pkl | c86587713a85feb89703b9f68fc6930d43000e341c7b97ca964d9400127359bf |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/clean2\buildup\cp_students_rk_val.pkl` | 444521 | 2026-11-13 10:54 | .pkl | 6c86b5483b84a9fea1cf4552146bd162f2fa23177bc3e3ccd831b1b83f62f2c9 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/clean2\shaspec\pf_s0_test_adabn.npz` | 864740 | 2026-11-13 10:59 | .npz | af7898dd63e2a76280df7eeaf9ac2378ef33bdf5ad778895ceca60a2fd52c670 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/clean2\shaspec\pf_s0_test_raw.npz` | 864740 | 2026-11-13 10:57 | .npz | 95b89e40c79cb386ec23f54a9a13c07b7afdc630e0cf24298fdfac906e4891b1 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/clean2\shaspec\pf_s0_val_adabn.npz` | 421220 | 2026-11-13 10:58 | .npz | a80dba2315ecbdd38c80b0c9daad57f1b971e049cd6ad7b395a80aa2a103afc1 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/clean2\shaspec\pf_s0_val_raw.npz` | 421220 | 2026-11-13 10:56 | .npz | 22051673b79497573642fe47b5354f5eabb076d7f8d9d648921c98b27363352c |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/clean2\shaspec\pf_s1_test_adabn.npz` | 864740 | 2026-11-13 11:01 | .npz | 82559638f78f16fa322c753bdc6b434eb4fd70cefe2a89ff70efd4ed1d679147 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/clean2\shaspec\pf_s1_test_raw.npz` | 864740 | 2026-11-13 11:00 | .npz | 524f19db3ca40db474e60d0a2556a7856fb2ad74d42820b08825c63e736704ec |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/clean2\shaspec\pf_s1_val_adabn.npz` | 421220 | 2026-11-13 11:01 | .npz | 4132c413e638315b7dff689c492894ec27922be5046c15e1d2662ef76faeb6df |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/clean2\shaspec\pf_s1_val_raw.npz` | 421220 | 2026-11-13 10:59 | .npz | e74ccf6f55abf6f2654f1f860349df3d8adf1b24ac315aa99522e08951d89c2f |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/clean2\w0\kps_te1.pkl` | 209355 | 2026-11-13 11:16 | .pkl | 9062b0fa6cd45688fc11cbb441c9d81a0c88471d8a8c1d23ec3798ea4781c59c |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/clean2\w0\kps_tr1.pkl` | 616223 | 2026-11-13 11:16 | .pkl | 820b0bb2ef3a6bd21940616de219d39d735f8539c20994594de5033b50f359c8 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/clean2\w0\kps_val1.pkl` | 110298 | 2026-11-13 11:16 | .pkl | da1ce186f10b1123f4b2656f76675db6a2f538dd1c4d4fa5c801ab46cb4f478a |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/clean2/w0/ou/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/clean2/w0/preds/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/clean2/w0/recs/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/clean2/w0/ref/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/clean2\w0\work\oc_te1.pkl` | 1707425 | 2026-11-13 11:16 | .pkl | bf3e21e180d2eebc48a3d57a843983e32a64e9f786f7fee3d4f5af8eececb7ed |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/clean2\w0\work\oc_tr1.pkl` | 4631227 | 2026-11-13 11:16 | .pkl | 1f5fd36e5b2e1a811c5c2c4fa5a734a53f7d22b28065b9f8493439b80a5c32f8 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/clean2\w0\work\oc_val1.pkl` | 839492 | 2026-11-13 11:16 | .pkl | 4ed7912b9fdc5c8bdc072cc8e63f45d8e72b34ace7acd4276040b868477b4e64 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/clean3\fig_qual.png` | 1289249 | 2026-11-13 12:45 | .png | 7a575882720e50600866033388a9594756287f7b23305d08ba7dfb6b6c411427 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/clean3\fig_qual_main.png` | 537162 | 2026-11-13 12:52 | .png | 7c5b73a8810c8e7582c88737286a23e7460fa3c7ed0ab7f538373133cfb46a85 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/clean3\nomir\kps_te1.pkl` | 209355 | 2026-11-13 12:54 | .pkl | 9062b0fa6cd45688fc11cbb441c9d81a0c88471d8a8c1d23ec3798ea4781c59c |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/clean3\nomir\kps_tr1.pkl` | 616223 | 2026-11-13 12:54 | .pkl | 820b0bb2ef3a6bd21940616de219d39d735f8539c20994594de5033b50f359c8 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/clean3\nomir\kps_val1.pkl` | 110298 | 2026-11-13 12:54 | .pkl | da1ce186f10b1123f4b2656f76675db6a2f538dd1c4d4fa5c801ab46cb4f478a |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/clean3/nomir/ou/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/clean3/nomir/preds/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/clean3/nomir/recs/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/clean3/nomir/ref/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/clean3\nomir\work\oc_te1.pkl` | 1707425 | 2026-11-13 12:54 | .pkl | bf3e21e180d2eebc48a3d57a843983e32a64e9f786f7fee3d4f5af8eececb7ed |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/clean3\nomir\work\oc_tr1.pkl` | 4631227 | 2026-11-13 12:54 | .pkl | 1f5fd36e5b2e1a811c5c2c4fa5a734a53f7d22b28065b9f8493439b80a5c32f8 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/clean3\nomir\work\oc_val1.pkl` | 839492 | 2026-11-13 12:54 | .pkl | 4ed7912b9fdc5c8bdc072cc8e63f45d8e72b34ace7acd4276040b868477b4e64 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/clean3\nomir\work\pseudo_tr1.npz` | 5574350 | 2026-11-13 12:54 | .npz | 896e68a62b5d364efaae0ee4c77e396bbac6cfe1513238199695085432410934 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/clean3\qual\cache_anti.pkl` | 1691147 | 2026-11-17 17:35 | .pkl | 13e76c3b2e9e489dbc6c0df5c15bdfbbbb339ca8b8a97c454d78015ed8f5c3ca |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/clean3\qual\cache_mav.pkl` | 3726328 | 2026-11-17 17:35 | .pkl | 363f41e653eaa9bbcaa9b6f4ed71ecfe9cf752c6491a255cf017cb91d88cdb03 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/clean3\qual\cand2.png` | 440277 | 2026-11-17 17:27 | .png | 72e0465e880b464517a6e4cfa1c940744f560ca4c0fd006a3a05496f95291355 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/clean3\qual\candB_anti0.png` | 73016 | 2026-11-17 17:36 | .png | f1eb15a92d8f6803cfd6f8e860a8fe9b6225e46cd747c6d34ca86402d67f53a9 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/clean3\qual\candB_anti1.png` | 28048 | 2026-11-17 17:37 | .png | 22e117556f9385734d9b60149d5ea7da0910cd4bc711b15d1bd4716ca994f200 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/clean3\qual\candB_mav0.png` | 86687 | 2026-11-17 17:36 | .png | e388618ae7abf1cf3743fe4faeff96996de6a10f456edfc14695bddae16e0fc9 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/clean3\qual\candB_mav1.png` | 394950 | 2026-11-17 17:36 | .png | 641bbdc2f2708392dfe2a6c64125b19199e2a5ff0ea679f5f7c6c155e58e0ba7 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/clean3\qual\cand_anti.png` | 106773 | 2026-11-17 17:28 | .png | cb0bab20355854bc8a1ab0a2e10ba1bf92e20bf7d7616c8a2d6ce1ad722e4ea3 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/clean3\qual\cand_mavic.png` | 153005 | 2026-11-17 17:21 | .png | da9eec5234edfd13060225b9351f34457e963af71044c87c5ca4e523fd0b3b3d |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/clean3\qual\fig_qual_main_cmp1117.png` | 954127 | 2026-11-17 17:28 | .png | c61d006da0c0af1bc688b5dd5d4ce029a9b740607769d1b42f368720589b0da6 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/clean3\qual\fig_qual_main_cmp4_1117.png` | 1579246 | 2026-11-17 17:45 | .png | 2b8c0ae64ca38f7b604375fe9147af6b8f65f5c5ab18a3264a9dbe689ce8b859 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/cleanfix/A2/in2mav/preds/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/cleanfix/A2/out2antifs/preds/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/cleanfix/A3/preds/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/cleanfix\A4\a4_ens.pkl` | 11989565 | 2026-11-14 15:01 | .pkl | aa4bf4c088333bbb445dedf2462cad8b3aedfff5bf96ee0d8944de7b9fcd5b46 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/cleanfix/B1_timing/out/ref/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/cleanfix\B2_etasq\feats_KPSs0.npz` | 80390440 | 2026-11-14 19:33 | .npz |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/cleanfix\B2_etasq\feats_KPSs0_smoke.npz` | 1042035 | 2026-11-14 14:45 | .npz | eafffbbfa150c456a98eaa79ddaad5582d235058d5a689dc27deb705886759d2 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/cleanfix\B2_etasq\feats_KPSs1.npz` | 77309006 | 2026-11-14 19:33 | .npz |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/cleanfix\B2_etasq\feats_KPSs2.npz` | 78630537 | 2026-11-14 19:34 | .npz |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/cleanfix\B3_noself\arm\kps_te1.pkl` | 209355 | 2026-11-14 14:47 | .pkl | 9062b0fa6cd45688fc11cbb441c9d81a0c88471d8a8c1d23ec3798ea4781c59c |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/cleanfix\B3_noself\arm\kps_tr1.pkl` | 616223 | 2026-11-14 14:47 | .pkl | 820b0bb2ef3a6bd21940616de219d39d735f8539c20994594de5033b50f359c8 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/cleanfix\B3_noself\arm\kps_val1.pkl` | 110298 | 2026-11-14 14:47 | .pkl | da1ce186f10b1123f4b2656f76675db6a2f538dd1c4d4fa5c801ab46cb4f478a |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/cleanfix/B3_noself/arm/ou/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/cleanfix/B3_noself/arm/preds/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/cleanfix/B3_noself/arm/recs/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/cleanfix/B3_noself/arm/ref/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/cleanfix\B3_noself\arm\work\oc_te1.pkl` | 1707425 | 2026-11-14 14:47 | .pkl | bf3e21e180d2eebc48a3d57a843983e32a64e9f786f7fee3d4f5af8eececb7ed |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/cleanfix\B3_noself\arm\work\oc_tr1.pkl` | 4631227 | 2026-11-14 14:47 | .pkl | 1f5fd36e5b2e1a811c5c2c4fa5a734a53f7d22b28065b9f8493439b80a5c32f8 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/cleanfix\B3_noself\arm\work\oc_val1.pkl` | 839492 | 2026-11-14 14:47 | .pkl | 4ed7912b9fdc5c8bdc072cc8e63f45d8e72b34ace7acd4276040b868477b4e64 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/cleanfix\B4_orient\nocomp\kps_te1.pkl` | 209355 | 2026-11-14 14:47 | .pkl | 9062b0fa6cd45688fc11cbb441c9d81a0c88471d8a8c1d23ec3798ea4781c59c |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/cleanfix\B4_orient\nocomp\kps_tr1.pkl` | 616223 | 2026-11-14 14:47 | .pkl | 820b0bb2ef3a6bd21940616de219d39d735f8539c20994594de5033b50f359c8 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/cleanfix\B4_orient\nocomp\kps_val1.pkl` | 110298 | 2026-11-14 14:47 | .pkl | da1ce186f10b1123f4b2656f76675db6a2f538dd1c4d4fa5c801ab46cb4f478a |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/cleanfix/B4_orient/nocomp/ou/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/cleanfix/B4_orient/nocomp/preds/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/cleanfix/B4_orient/nocomp/recs/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/cleanfix/B4_orient/nocomp/ref/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/cleanfix\B4_orient\nocomp\work\oc_te1.pkl` | 1707425 | 2026-11-14 14:47 | .pkl | bf3e21e180d2eebc48a3d57a843983e32a64e9f786f7fee3d4f5af8eececb7ed |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/cleanfix\B4_orient\nocomp\work\oc_tr1.pkl` | 4631227 | 2026-11-14 14:47 | .pkl | 1f5fd36e5b2e1a811c5c2c4fa5a734a53f7d22b28065b9f8493439b80a5c32f8 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/cleanfix\B4_orient\nocomp\work\oc_val1.pkl` | 839492 | 2026-11-14 14:47 | .pkl | 4ed7912b9fdc5c8bdc072cc8e63f45d8e72b34ace7acd4276040b868477b4e64 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/cleanfix\B4_orient\nocomp\work\pseudo_tr1.npz` | 5574350 | 2026-11-14 19:01 | .npz | 896e68a62b5d364efaae0ee4c77e396bbac6cfe1513238199695085432410934 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/cleanfix\B4_orient\nosym\kps_te1.pkl` | 209355 | 2026-11-14 14:47 | .pkl | 9062b0fa6cd45688fc11cbb441c9d81a0c88471d8a8c1d23ec3798ea4781c59c |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/cleanfix\B4_orient\nosym\kps_tr1.pkl` | 616223 | 2026-11-14 14:47 | .pkl | 820b0bb2ef3a6bd21940616de219d39d735f8539c20994594de5033b50f359c8 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/cleanfix\B4_orient\nosym\kps_val1.pkl` | 110298 | 2026-11-14 14:47 | .pkl | da1ce186f10b1123f4b2656f76675db6a2f538dd1c4d4fa5c801ab46cb4f478a |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/cleanfix/B4_orient/nosym/ou/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/cleanfix/B4_orient/nosym/preds/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/cleanfix/B4_orient/nosym/recs/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/cleanfix/B4_orient/nosym/ref/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/cleanfix\B4_orient\nosym\work\oc_te1.pkl` | 1707425 | 2026-11-14 14:47 | .pkl | bf3e21e180d2eebc48a3d57a843983e32a64e9f786f7fee3d4f5af8eececb7ed |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/cleanfix\B4_orient\nosym\work\oc_tr1.pkl` | 4631227 | 2026-11-14 14:47 | .pkl | 1f5fd36e5b2e1a811c5c2c4fa5a734a53f7d22b28065b9f8493439b80a5c32f8 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/cleanfix\B4_orient\nosym\work\oc_val1.pkl` | 839492 | 2026-11-14 14:47 | .pkl | 4ed7912b9fdc5c8bdc072cc8e63f45d8e72b34ace7acd4276040b868477b4e64 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/cleanfix\B4_orient\nosym\work\pseudo_tr1.npz` | 5574350 | 2026-11-14 20:10 | .npz | 896e68a62b5d364efaae0ee4c77e396bbac6cfe1513238199695085432410934 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/cleanfix\B5_bgrel\ctx_examples.jpg` | 402172 | 2026-11-14 19:34 | .jpg | 6a4e80d7f82376d07e287cabeeb28ee1551e21987147b9c29b9c8f9209a49b3a |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/cleanfix\B5_bgrel\kp_KZ_s0.npz` | 213414 | 2026-11-14 19:35 | .npz | 6ae4d1c0c75a1400fa8b5361616b6033482f38057879d5dd50fe68a2cdb2dcc1 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/cleanfix\B5_bgrel\kp_KZ_s1.npz` | 213414 | 2026-11-14 19:35 | .npz | c913022e169eeba7fea1f857e4955d62d6b4d75b77670d764d46501fd980591f |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/cleanfix\B5_bgrel\kp_KZ_s2.npz` | 213414 | 2026-11-14 19:36 | .npz | b33dc658c805c5252303ed78a29213f3c010a67b630b8f1756b79edfc4c9fbb5 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/cleanfix\B5_bgrel\kp_KZ_s3.npz` | 213414 | 2026-11-14 19:37 | .npz | b7c86708f4220fc14f5759d50ca5cfb7b82317250577b5211feb24f0645a88eb |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/cleanfix\B5_bgrel\kp_KZ_s4.npz` | 213414 | 2026-11-14 19:37 | .npz | c00f566e04ad90c2b78c67d4d4a0810c0b8103746a76f6a9e31921f3960408e7 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/cleanfix/B5_bgrel/smoke/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/cleanfix\B6_gtceil\arm\kps_te1.pkl` | 209355 | 2026-11-14 14:47 | .pkl | 9062b0fa6cd45688fc11cbb441c9d81a0c88471d8a8c1d23ec3798ea4781c59c |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/cleanfix\B6_gtceil\arm\kps_tr1.pkl` | 616223 | 2026-11-14 14:47 | .pkl | 820b0bb2ef3a6bd21940616de219d39d735f8539c20994594de5033b50f359c8 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/cleanfix\B6_gtceil\arm\kps_val1.pkl` | 110298 | 2026-11-14 14:47 | .pkl | da1ce186f10b1123f4b2656f76675db6a2f538dd1c4d4fa5c801ab46cb4f478a |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/cleanfix/B6_gtceil/arm/ou/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/cleanfix/B6_gtceil/arm/preds/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/cleanfix/B6_gtceil/arm/recs/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/cleanfix/B6_gtceil/arm/ref/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/cleanfix\B6_gtceil\arm\work\oc_te1.pkl` | 1707425 | 2026-11-14 14:47 | .pkl | bf3e21e180d2eebc48a3d57a843983e32a64e9f786f7fee3d4f5af8eececb7ed |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/cleanfix\B6_gtceil\arm\work\oc_tr1.pkl` | 4631227 | 2026-11-14 14:47 | .pkl | 1f5fd36e5b2e1a811c5c2c4fa5a734a53f7d22b28065b9f8493439b80a5c32f8 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/cleanfix\B6_gtceil\arm\work\oc_val1.pkl` | 839492 | 2026-11-14 14:47 | .pkl | 4ed7912b9fdc5c8bdc072cc8e63f45d8e72b34ace7acd4276040b868477b4e64 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/cleanfix\B6_gtceil\arm\work\pseudo_gt.npz` | 5574350 | 2026-11-14 14:47 | .npz | d4425447b16abe928feeab58013b0551b177f1bd92e577e662422f9db7f2f504 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/corrupt_1117/out2antifs/clean/ou/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/corrupt_1117/out2antifs/clean/preds/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/corrupt_1117/out2antifs/clean/recs/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/corrupt_1117/out2antifs/clean/ref/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/cpu5\anti_b_det1_test.pkl` | 445592 | 2026-11-14 02:57 | .pkl | 53209e614ae95a28816fd208df544155ae1676217fadb6ee9b2ee060a5595b40 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/cpu5\anti_b_det1_val.pkl` | 195154 | 2026-11-14 02:57 | .pkl | 0b28a58064566d410eefa05eede36f335679adf9d0c9cff22d27e79cd64f9d31 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/cpu5\anti_c_students_test.pkl` | 445592 | 2026-11-14 02:57 | .pkl | 6ead80fd81033b8bafbe06cd8c67c43591d49d402bb64d83340e45629d733615 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/cpu5\anti_c_students_val.pkl` | 195154 | 2026-11-14 02:57 | .pkl | 2e647ad2d3a419409ee0ce1e0b4a41b192aae5d3822c7b0929b51aeb084429c5 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/cpu5\anti_cp_students_rk_test.pkl` | 445592 | 2026-11-14 02:57 | .pkl | 82f58a0a18e0db622385cd32d11633487da59daf63479ae4c5fb1c6e572b7dde |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/cpu5\anti_cp_students_rk_val.pkl` | 195154 | 2026-11-14 02:57 | .pkl | c154787e1418323acbf5e38fa384013eac88b18d267ada31db0af9b95af85462 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/figdata\testgap\st2_s5_det0_test.pkl` | 343782 | 2026-11-14 17:22 | .pkl | cfc2bfef2d13b851ab87db7a0c53a97bc8c3bcbcbaf8bc91db1dd54741f1735e |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/figdata\testgap\st2_s6_det0_test.pkl` | 343782 | 2026-11-14 17:24 | .pkl | afa642554f6b17a81e07beee80632f314fdc3855754b0c68e94a4a9b8b0edb73 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads\H1\mav_s0\kps_te1.pkl` | 209355 | 2026-11-15 17:13 | .pkl | c56319214cd89f01155f06e730a96dcc99e13e337eec9745fe62f442221d14bb |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads\H1\mav_s0\kps_tr1.pkl` | 616223 | 2026-11-15 17:11 | .pkl | 347311edb6c1a6cf98f81cb138bc7cafd8a583464037e4adf0a1a68f2f34ba72 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads\H1\mav_s0\kps_val1.pkl` | 110298 | 2026-11-15 17:12 | .pkl | 12833751018e8a58ae8d693b4fb7c842845142d645048e878ea3bd12ed41f51e |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads/H1/mav_s0/ou/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads/H1/mav_s0/preds/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads/H1/mav_s0/recs/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads/H1/mav_s0/ref/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads\H1\mav_s0\wsim_cache.pkl` | 1912979 | 2026-11-15 00:37 | .pkl | 3b27c15c20c112b28d385f9c6b7506d692c5a90850de16d51afc5ec24ada878c |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads\H1\mav_s0\wsim_read.pkl` | 10352764 | 2026-11-15 17:35 | .pkl | 83e383622180a12793f0f73058a955df0d4031b2c93d59305de4abb05d69a976 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads\H1\mav_s10\kps_te1.pkl` | 209355 | 2026-11-17 09:26 | .pkl | 376651a4f82797679c08e76832b0a3b41938ed7a26fd808b8927c6f77f705689 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads\H1\mav_s10\kps_tr1.pkl` | 616223 | 2026-11-17 09:23 | .pkl | cc2418c312a97ab43cc5eabf513b0e88a811adcdb2e0996c0cee420d9cbbb6d3 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads\H1\mav_s10\kps_val1.pkl` | 110298 | 2026-11-17 09:24 | .pkl | 88ee34cb49e35259f6492d4d68bab40b023193477c9cd5c27dbf6c0a7dc36594 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads/H1/mav_s10/ou/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads/H1/mav_s10/preds/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads/H1/mav_s10/recs/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads/H1/mav_s10/ref/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads\H1\mav_s10\wsim_cache.pkl` | 1912979 | 2026-11-16 12:28 | .pkl | 3b27c15c20c112b28d385f9c6b7506d692c5a90850de16d51afc5ec24ada878c |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads\H1c\mav_s0\kps_te1.pkl` | 209355 | 2026-11-14 21:48 | .pkl | 9062b0fa6cd45688fc11cbb441c9d81a0c88471d8a8c1d23ec3798ea4781c59c |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads\H1c\mav_s0\kps_tr1.pkl` | 616223 | 2026-11-14 21:48 | .pkl | 820b0bb2ef3a6bd21940616de219d39d735f8539c20994594de5033b50f359c8 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads\H1c\mav_s0\kps_val1.pkl` | 110298 | 2026-11-14 21:48 | .pkl | da1ce186f10b1123f4b2656f76675db6a2f538dd1c4d4fa5c801ab46cb4f478a |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads/H1c/mav_s0/ou/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads/H1c/mav_s0/preds/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads/H1c/mav_s0/recs/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads/H1c/mav_s0/ref/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads\H1c\mav_s0\work\oc_te1.pkl` | 1707425 | 2026-11-14 21:48 | .pkl | bf3e21e180d2eebc48a3d57a843983e32a64e9f786f7fee3d4f5af8eececb7ed |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads\H1c\mav_s0\work\oc_tr1.pkl` | 4631227 | 2026-11-14 21:48 | .pkl | 1f5fd36e5b2e1a811c5c2c4fa5a734a53f7d22b28065b9f8493439b80a5c32f8 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads\H1c\mav_s0\work\oc_val1.pkl` | 839492 | 2026-11-14 21:48 | .pkl | 4ed7912b9fdc5c8bdc072cc8e63f45d8e72b34ace7acd4276040b868477b4e64 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads\H1c\mav_s10\kps_te1.pkl` | 209355 | 2026-11-16 07:10 | .pkl | 5643b8e63d5ee793f382b3231cb0ceedba28c7f1c93f7c1ae45c6c0fbac8fb93 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads\H1c\mav_s10\kps_tr1.pkl` | 616223 | 2026-11-16 07:10 | .pkl | aa90395c7519b78721e65d5e28119ee2e6d7619300be670aa1cb323ccaf1680e |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads\H1c\mav_s10\kps_val1.pkl` | 110298 | 2026-11-16 07:10 | .pkl | 56ee82b406127dba198fb20d4732cbf0c202c365b1a3f6a8377329b104d3f37e |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads/H1c/mav_s10/ou/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads/H1c/mav_s10/preds/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads/H1c/mav_s10/recs/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads/H1c/mav_s10/ref/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads\H1c\mav_s10\work\oc_te1.pkl` | 1707425 | 2026-11-16 07:10 | .pkl | ac54c31b27f9d16aac02f1142b3e46a883e8e176588fcc5d119263f1f7f2430f |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads\H1c\mav_s10\work\oc_tr1.pkl` | 4631227 | 2026-11-16 07:10 | .pkl | 748fea8d603c77ae4174aa47a64cb1026d0af02e84e31ea617f6b926e0619f0e |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads\H1c\mav_s10\work\oc_val1.pkl` | 839492 | 2026-11-16 07:10 | .pkl | 09d5da21a98d9b710c62cb525ccf226ea05962c8c692fe149d1f769e33428021 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads\H1c\mav_s20\kps_te1.pkl` | 209355 | 2026-11-16 10:28 | .pkl | e26b0bd9cfb051fa807bccf680caea54c2c3e14993ed24586facc9639a87024e |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads\H1c\mav_s20\kps_tr1.pkl` | 616223 | 2026-11-16 10:28 | .pkl | 0345ce60d61d9c0333f549c2c424781ac57c683bf1ccb9fc586557dbfc0d9dea |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads\H1c\mav_s20\kps_val1.pkl` | 110298 | 2026-11-16 10:28 | .pkl | 7ddb73da9639923326959623c46384c53311075c35bbb55497d41413d5c7de16 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads/H1c/mav_s20/ou/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads/H1c/mav_s20/preds/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads/H1c/mav_s20/recs/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads/H1c/mav_s20/ref/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads\H1c\mav_s20\work\oc_te1.pkl` | 1707425 | 2026-11-16 10:28 | .pkl | 74c25b66feaf6aa523b055acd46b33bfc7ee5cbc0a9d02027b0c4f28ade512af |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads\H1c\mav_s20\work\oc_tr1.pkl` | 4631227 | 2026-11-16 10:28 | .pkl | e3183fd06578d5b509280a6e076a2089677c213bf0032c2526c31db985b9a6e9 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads\H1c\mav_s20\work\oc_val1.pkl` | 839492 | 2026-11-16 10:28 | .pkl | b3adb3318b740f196329b647ad76ee026cf136e733d2b109fc6668e41270dda0 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads\H2\mav_s0\kps_te1.pkl` | 209355 | 2026-11-15 01:54 | .pkl | 7bdabf498f5bc8713cc49a239ac2b42bce3657dbdb665787eaf0cf0d1e6f61ea |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads\H2\mav_s0\kps_tr1.pkl` | 616223 | 2026-11-15 01:53 | .pkl | 133a5d22e4f17c935a1ece1b68c6ea8061122412a59ced1e65a4cc872e278f47 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads\H2\mav_s0\kps_val1.pkl` | 110298 | 2026-11-15 01:53 | .pkl | d47d557d6f73bf01c5bc79a142f9470addc0c1c3a999908fc26c827500d4bc7c |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads/H2/mav_s0/ou/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads/H2/mav_s0/preds/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads/H2/mav_s0/recs/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads/H2/mav_s0/ref/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads\H2\mav_s0\wsim_cache.pkl` | 1912979 | 2026-11-14 22:58 | .pkl | 3b27c15c20c112b28d385f9c6b7506d692c5a90850de16d51afc5ec24ada878c |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads\H2\mav_s0\wsim_read.pkl` | 10352764 | 2026-11-15 15:04 | .pkl | 7c83e36b885a6a74321d67e637ad0460ba5613b48701ebaa7f6ce6f0d68066cd |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads\H2\mav_s10\kps_te1.pkl` | 209355 | 2026-11-16 11:05 | .pkl | 45ec45a22a1ebe951c8241c19fbfa9f28199256795154f585b890a1407a78708 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads\H2\mav_s10\kps_tr1.pkl` | 616223 | 2026-11-16 11:04 | .pkl | dff62d9f082df5962d803a69d4324b90f4aad1aa9942bf03c8e7acde4a34a2e1 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads\H2\mav_s10\kps_val1.pkl` | 110298 | 2026-11-16 11:04 | .pkl | dba77121a046f8e4383bdede7792c7fead4547827a2d9de9019c59691b696924 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads/H2/mav_s10/ou/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads/H2/mav_s10/preds/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads/H2/mav_s10/recs/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads/H2/mav_s10/ref/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads\H2\mav_s10\wsim_cache.pkl` | 1912979 | 2026-11-16 07:42 | .pkl | 3b27c15c20c112b28d385f9c6b7506d692c5a90850de16d51afc5ec24ada878c |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads\H2\mav_s10\wsim_read.pkl` | 10352768 | 2026-11-16 11:38 | .pkl | 3714bd2758375fc0410a857f8fe6b5e48dbd6f301238658dd7b6dcbc1e38aa7e |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads\H2\mav_s20\logs\ori_infer.log.try` | 0 | 2026-11-17 21:08 | .try |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads/H2/mav_s20/ou/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads/H2/mav_s20/preds/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads/H2/mav_s20/recs/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads/H2/mav_s20/ref/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads\H2\mav_s20\wsim_cache.pkl` | 1912979 | 2026-11-17 12:19 | .pkl | 3b27c15c20c112b28d385f9c6b7506d692c5a90850de16d51afc5ec24ada878c |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads\H3\mav_s0\kps_te1.pkl` | 209355 | 2026-11-15 23:59 | .pkl | a5fac1a146eb43d8515b5337653d8e23762ceb84e58d2350ffd873957899fb0c |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads\H3\mav_s0\kps_tr1.pkl` | 616223 | 2026-11-15 23:59 | .pkl | 029a745793e615c929577691d011b5aea2b783e39e6c153f72d57cfd32ee9704 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads\H3\mav_s0\kps_val1.pkl` | 110298 | 2026-11-15 23:59 | .pkl | ada33c4efe3b6d718a6dc8c2745d7ed23f435281d49cf733e80ffa5f07973645 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads/H3/mav_s0/ou/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads/H3/mav_s0/preds/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads/H3/mav_s0/recs/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads/H3/mav_s0/ref/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads\H3\mav_s0\wsim_cache.pkl` | 1912979 | 2026-11-15 19:11 | .pkl | 3b27c15c20c112b28d385f9c6b7506d692c5a90850de16d51afc5ec24ada878c |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads\H3\mav_s0\wsim_read.pkl` | 10352764 | 2026-11-16 01:00 | .pkl | e9b5476a0bca2af85f3204368dad36ff92154560e7a2626cd1ae2a2a17e9565b |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads/H3/mav_s10/ou/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads/H3/mav_s10/preds/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads/H3/mav_s10/recs/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads/H3/mav_s10/ref/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads\H3\mav_s10\wsim_cache.pkl` | 1912979 | 2026-11-16 17:32 | .pkl | 3b27c15c20c112b28d385f9c6b7506d692c5a90850de16d51afc5ec24ada878c |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads\H4\antifs_s0\kps_te1.pkl` | 70185 | 2026-11-17 04:39 | .pkl | 04a1d3b583f44dd42ac055dfd18ea4e239858ed0ce70b541ef01112b65c9be52 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads\H4\antifs_s0\kps_tr1.pkl` | 95245 | 2026-11-17 04:39 | .pkl | 6a658b3667380b92ede2c8ad3aea896ebd5483bb158b7a59f54faa319281b74a |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads\H4\antifs_s0\kps_val1.pkl` | 30842 | 2026-11-17 04:39 | .pkl | a68623ada3cc328e6633e08556c31176ee350d50bcedf173f5788a1fea8cee09 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads/H4/antifs_s0/ou/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads/H4/antifs_s0/preds/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads/H4/antifs_s0/recs/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads/H4/antifs_s0/ref/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads\H4\antifs_s10\kps_te1.pkl` | 70185 | 2026-11-17 05:37 | .pkl | f3d6ca6021c7a5c1109658e17dab6f241f062dd827c0dc9407c4a2d7ee87e46a |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads\H4\antifs_s10\kps_tr1.pkl` | 95245 | 2026-11-17 05:37 | .pkl | b23b93abfb778f31e198a061c78904b040859dfaa87630bb3a7b343a8bcfa192 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads\H4\antifs_s10\kps_val1.pkl` | 30842 | 2026-11-17 05:37 | .pkl | ed1621505d5c9d6eb5acb25e1f5768253a1ed53c135c9fa444d3c32100afcfcf |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads/H4/antifs_s10/ou/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads/H4/antifs_s10/preds/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads/H4/antifs_s10/recs/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads/H4/antifs_s10/ref/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads\H4\antifs_s20\kps_te1.pkl` | 70185 | 2026-11-17 06:36 | .pkl | bddd8756f6c1074b01dc58df698a5029b5372ec1e35237c645fdfa1707c65a72 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads\H4\antifs_s20\kps_tr1.pkl` | 95245 | 2026-11-17 06:36 | .pkl | c1f1d99acf7429776d277559eb446911013cb69d84c34a8ff175d1853b0b4f36 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads\H4\antifs_s20\kps_val1.pkl` | 30842 | 2026-11-17 06:36 | .pkl | c2b8d9c0df9732d2aecffe2652cd7bb7a00d1bdf09c4a6d267471b372d61efdd |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads/H4/antifs_s20/ou/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads/H4/antifs_s20/preds/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads/H4/antifs_s20/recs/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads/H4/antifs_s20/ref/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads\H4\mav_s0\kps_te1.pkl` | 209355 | 2026-11-14 21:38 | .pkl | 9062b0fa6cd45688fc11cbb441c9d81a0c88471d8a8c1d23ec3798ea4781c59c |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads\H4\mav_s0\kps_tr1.pkl` | 616223 | 2026-11-14 21:38 | .pkl | 820b0bb2ef3a6bd21940616de219d39d735f8539c20994594de5033b50f359c8 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads\H4\mav_s0\kps_val1.pkl` | 110298 | 2026-11-14 21:38 | .pkl | da1ce186f10b1123f4b2656f76675db6a2f538dd1c4d4fa5c801ab46cb4f478a |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads/H4/mav_s0/ou/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads/H4/mav_s0/preds/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads/H4/mav_s0/recs/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads/H4/mav_s0/ref/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads\H4\mav_s10\kps_te1.pkl` | 209355 | 2026-11-16 06:31 | .pkl | 5643b8e63d5ee793f382b3231cb0ceedba28c7f1c93f7c1ae45c6c0fbac8fb93 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads\H4\mav_s10\kps_tr1.pkl` | 616223 | 2026-11-16 06:31 | .pkl | aa90395c7519b78721e65d5e28119ee2e6d7619300be670aa1cb323ccaf1680e |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads\H4\mav_s10\kps_val1.pkl` | 110298 | 2026-11-16 06:31 | .pkl | 56ee82b406127dba198fb20d4732cbf0c202c365b1a3f6a8377329b104d3f37e |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads/H4/mav_s10/ou/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads/H4/mav_s10/preds/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads/H4/mav_s10/recs/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads/H4/mav_s10/ref/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads\H4\mav_s20\kps_te1.pkl` | 209355 | 2026-11-16 09:32 | .pkl | e26b0bd9cfb051fa807bccf680caea54c2c3e14993ed24586facc9639a87024e |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads\H4\mav_s20\kps_tr1.pkl` | 616223 | 2026-11-16 09:32 | .pkl | 0345ce60d61d9c0333f549c2c424781ac57c683bf1ccb9fc586557dbfc0d9dea |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads\H4\mav_s20\kps_val1.pkl` | 110298 | 2026-11-16 09:32 | .pkl | 7ddb73da9639923326959623c46384c53311075c35bbb55497d41413d5c7de16 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads/H4/mav_s20/ou/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads/H4/mav_s20/preds/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads/H4/mav_s20/recs/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads/H4/mav_s20/ref/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads\H5\antifs_s0\kps_te1.pkl` | 70185 | 2026-11-17 05:15 | .pkl | 04a1d3b583f44dd42ac055dfd18ea4e239858ed0ce70b541ef01112b65c9be52 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads\H5\antifs_s0\kps_tr1.pkl` | 95245 | 2026-11-17 05:15 | .pkl | 6a658b3667380b92ede2c8ad3aea896ebd5483bb158b7a59f54faa319281b74a |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads\H5\antifs_s0\kps_val1.pkl` | 30842 | 2026-11-17 05:15 | .pkl | a68623ada3cc328e6633e08556c31176ee350d50bcedf173f5788a1fea8cee09 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads/H5/antifs_s0/ou/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads/H5/antifs_s0/preds/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads/H5/antifs_s0/recs/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads/H5/antifs_s0/ref/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads\H5\antifs_s10\kps_te1.pkl` | 70185 | 2026-11-17 05:58 | .pkl | f3d6ca6021c7a5c1109658e17dab6f241f062dd827c0dc9407c4a2d7ee87e46a |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads\H5\antifs_s10\kps_tr1.pkl` | 95245 | 2026-11-17 05:58 | .pkl | b23b93abfb778f31e198a061c78904b040859dfaa87630bb3a7b343a8bcfa192 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads\H5\antifs_s10\kps_val1.pkl` | 30842 | 2026-11-17 05:58 | .pkl | ed1621505d5c9d6eb5acb25e1f5768253a1ed53c135c9fa444d3c32100afcfcf |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads/H5/antifs_s10/ou/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads/H5/antifs_s10/preds/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads/H5/antifs_s10/recs/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads/H5/antifs_s10/ref/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads\H5\antifs_s20\kps_te1.pkl` | 70185 | 2026-11-17 06:43 | .pkl | bddd8756f6c1074b01dc58df698a5029b5372ec1e35237c645fdfa1707c65a72 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads\H5\antifs_s20\kps_tr1.pkl` | 95245 | 2026-11-17 06:43 | .pkl | c1f1d99acf7429776d277559eb446911013cb69d84c34a8ff175d1853b0b4f36 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads\H5\antifs_s20\kps_val1.pkl` | 30842 | 2026-11-17 06:43 | .pkl | c2b8d9c0df9732d2aecffe2652cd7bb7a00d1bdf09c4a6d267471b372d61efdd |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads/H5/antifs_s20/ou/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads/H5/antifs_s20/preds/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads/H5/antifs_s20/recs/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads/H5/antifs_s20/ref/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads\H5\mav_s0\kps_te1.pkl` | 209355 | 2026-11-14 22:22 | .pkl | 9062b0fa6cd45688fc11cbb441c9d81a0c88471d8a8c1d23ec3798ea4781c59c |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads\H5\mav_s0\kps_tr1.pkl` | 616223 | 2026-11-14 22:22 | .pkl | 820b0bb2ef3a6bd21940616de219d39d735f8539c20994594de5033b50f359c8 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads\H5\mav_s0\kps_val1.pkl` | 110298 | 2026-11-14 22:22 | .pkl | da1ce186f10b1123f4b2656f76675db6a2f538dd1c4d4fa5c801ab46cb4f478a |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads/H5/mav_s0/ou/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads/H5/mav_s0/preds/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads/H5/mav_s0/recs/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads/H5/mav_s0/ref/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads\H5\mav_s10\kps_te1.pkl` | 209355 | 2026-11-16 07:11 | .pkl | 5643b8e63d5ee793f382b3231cb0ceedba28c7f1c93f7c1ae45c6c0fbac8fb93 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads\H5\mav_s10\kps_tr1.pkl` | 616223 | 2026-11-16 07:11 | .pkl | aa90395c7519b78721e65d5e28119ee2e6d7619300be670aa1cb323ccaf1680e |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads\H5\mav_s10\kps_val1.pkl` | 110298 | 2026-11-16 07:11 | .pkl | 56ee82b406127dba198fb20d4732cbf0c202c365b1a3f6a8377329b104d3f37e |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads/H5/mav_s10/ou/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads/H5/mav_s10/preds/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads/H5/mav_s10/recs/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads/H5/mav_s10/ref/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads\H5\mav_s20\kps_te1.pkl` | 209355 | 2026-11-16 10:28 | .pkl | e26b0bd9cfb051fa807bccf680caea54c2c3e14993ed24586facc9639a87024e |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads\H5\mav_s20\kps_tr1.pkl` | 616223 | 2026-11-16 10:28 | .pkl | 0345ce60d61d9c0333f549c2c424781ac57c683bf1ccb9fc586557dbfc0d9dea |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads\H5\mav_s20\kps_val1.pkl` | 110298 | 2026-11-16 10:28 | .pkl | 7ddb73da9639923326959623c46384c53311075c35bbb55497d41413d5c7de16 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads/H5/mav_s20/ou/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads/H5/mav_s20/preds/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads/H5/mav_s20/recs/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads/H5/mav_s20/ref/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads\H6\antifs_s0\kps_te1.pkl` | 70185 | 2026-11-17 04:46 | .pkl | 9a72632bb49851fdec0be4c389f6c56e6929e060c50b314cc53590db2bae120e |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads\H6\antifs_s0\kps_tr1.pkl` | 95216 | 2026-11-17 04:45 | .pkl | d3b29f5b15622fe8c6dc5bd846688095bf0a150a9415956db6668c3d191546e5 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads\H6\antifs_s0\kps_val1.pkl` | 30842 | 2026-11-17 04:45 | .pkl | 4e13c06a29b3c5a5c88c9892fbf287e174b2a1919cd141c2c22c05d70a3a1bf8 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads/H6/antifs_s0/ou/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads/H6/antifs_s0/preds/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads/H6/antifs_s0/recs/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads/H6/antifs_s0/ref/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads\H6\antifs_s10\kps_te1.pkl` | 70185 | 2026-11-17 04:50 | .pkl | ca224366be786080559042ff00482ee35ccb1af5366c56455cf09061a0eef6f5 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads\H6\antifs_s10\kps_tr1.pkl` | 95216 | 2026-11-17 04:48 | .pkl | 166adbeaf0f49652c20117ef8767f5ac89ca8b2c1ea2b925b8dda60dc9e29f3e |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads\H6\antifs_s10\kps_val1.pkl` | 30842 | 2026-11-17 04:49 | .pkl | 7b532391fa4cbbcbc6262664fa12cdc8cafce2f160d395faf698d6f95018c86f |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads/H6/antifs_s10/ou/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads/H6/antifs_s10/preds/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads/H6/antifs_s10/recs/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads/H6/antifs_s10/ref/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads\H6\antifs_s20\kps_te1.pkl` | 70185 | 2026-11-17 05:14 | .pkl | 45dcd5e584a05ac0b384055cfe1cc26b30246bf6bfd3d6e82a0a9e7651a2de19 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads\H6\antifs_s20\kps_tr1.pkl` | 95158 | 2026-11-17 05:13 | .pkl | 8fe5e64a075cd113ac0aca815740417f379383bab47a5034622c2508b7f7156c |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads\H6\antifs_s20\kps_val1.pkl` | 30842 | 2026-11-17 05:13 | .pkl | 143c8c0cfadfa19f1d6618a44faba76076e1760be27e9d1485478cb215b81ab5 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads/H6/antifs_s20/ou/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads/H6/antifs_s20/preds/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads/H6/antifs_s20/recs/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads/H6/antifs_s20/ref/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads\H6\mav_s0\kps_te1.pkl` | 209355 | 2026-11-14 21:48 | .pkl | 3c23968cf52a3f476f64571d332cf20fb5356f8560479c5a5640fcdad85f48b3 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads\H6\mav_s0\kps_tr1.pkl` | 616223 | 2026-11-14 21:46 | .pkl | b7f7981dcd26030ff9159e4d7a1afe2ed80cf6dcd558ed9988e43d68adab7d3f |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads\H6\mav_s0\kps_val1.pkl` | 110298 | 2026-11-14 21:46 | .pkl | 4595c52c790387f26c9c45dd1dfc52cae029f3d2d613035d30f837c1f152aced |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads/H6/mav_s0/ou/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads/H6/mav_s0/preds/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads/H6/mav_s0/recs/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads/H6/mav_s0/ref/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads\H6\mav_s10\kps_te1.pkl` | 209355 | 2026-11-16 07:10 | .pkl | bcb510c1259eac8d28e1994bcecc7d73dcdfd6126c281e9d093dfe2bf84f2f14 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads\H6\mav_s10\kps_tr1.pkl` | 616223 | 2026-11-16 07:09 | .pkl | 5b606c6e5fa740ac1ef1d20c1b70146dadd3b7d796ad29725f34fcd6d35b9533 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads\H6\mav_s10\kps_val1.pkl` | 110298 | 2026-11-16 07:09 | .pkl | e3e30e16b844cc00641778577b96b1a2328bd0e9fd2dc8f0117fcd2363f55865 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads/H6/mav_s10/ou/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads/H6/mav_s10/preds/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads/H6/mav_s10/recs/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads/H6/mav_s10/ref/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads\H6\mav_s20\kps_te1.pkl` | 209355 | 2026-11-16 10:27 | .pkl | 9e01e9b7c20667d737cf54f512e9b0cd8a0edebd037c88d32e20c5275c31ff28 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads\H6\mav_s20\kps_tr1.pkl` | 616223 | 2026-11-16 10:25 | .pkl | 406394b4b1e570f103dc133f693d8d2a97dcbc3d69ee519d72b0bbb6d0233fcb |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads\H6\mav_s20\kps_val1.pkl` | 110298 | 2026-11-16 10:26 | .pkl | 78c0ed5ba1b75990033b75bff972dac42a44b4a8c93690cab9048b8baba39d4c |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads/H6/mav_s20/ou/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads/H6/mav_s20/preds/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads/H6/mav_s20/recs/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads/H6/mav_s20/ref/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads\H7\mav_s10\kps_te1.pkl` | 209355 | 2026-11-16 02:24 | .pkl | 5643b8e63d5ee793f382b3231cb0ceedba28c7f1c93f7c1ae45c6c0fbac8fb93 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads\H7\mav_s10\kps_tr1.pkl` | 616223 | 2026-11-16 02:24 | .pkl | aa90395c7519b78721e65d5e28119ee2e6d7619300be670aa1cb323ccaf1680e |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads\H7\mav_s10\kps_val1.pkl` | 110298 | 2026-11-16 02:24 | .pkl | 56ee82b406127dba198fb20d4732cbf0c202c365b1a3f6a8377329b104d3f37e |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads/H7/mav_s10/ou/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads/H7/mav_s10/preds/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads/H7/mav_s10/recs/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads/H7/mav_s10/ref/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads\H7\mav_s10\work\oc_te1.pkl` | 1707425 | 2026-11-16 02:34 | .pkl | ac54c31b27f9d16aac02f1142b3e46a883e8e176588fcc5d119263f1f7f2430f |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads\H7\mav_s10\work\oc_tr1.pkl` | 4631227 | 2026-11-16 02:34 | .pkl | 748fea8d603c77ae4174aa47a64cb1026d0af02e84e31ea617f6b926e0619f0e |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads\H7\mav_s10\work\oc_val1.pkl` | 839492 | 2026-11-16 02:34 | .pkl | 09d5da21a98d9b710c62cb525ccf226ea05962c8c692fe149d1f769e33428021 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads\H7\mav_s20\kps_te1.pkl` | 209355 | 2026-11-16 04:34 | .pkl | e26b0bd9cfb051fa807bccf680caea54c2c3e14993ed24586facc9639a87024e |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads\H7\mav_s20\kps_tr1.pkl` | 616223 | 2026-11-16 04:34 | .pkl | 0345ce60d61d9c0333f549c2c424781ac57c683bf1ccb9fc586557dbfc0d9dea |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads\H7\mav_s20\kps_val1.pkl` | 110298 | 2026-11-16 04:34 | .pkl | 7ddb73da9639923326959623c46384c53311075c35bbb55497d41413d5c7de16 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads/H7/mav_s20/ou/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads/H7/mav_s20/preds/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads/H7/mav_s20/recs/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads/H7/mav_s20/ref/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads\H7\mav_s20\work\oc_te1.pkl` | 1707425 | 2026-11-16 04:39 | .pkl | 74c25b66feaf6aa523b055acd46b33bfc7ee5cbc0a9d02027b0c4f28ade512af |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads\H7\mav_s20\work\oc_tr1.pkl` | 4631227 | 2026-11-16 04:38 | .pkl | e3183fd06578d5b509280a6e076a2089677c213bf0032c2526c31db985b9a6e9 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads\H7\mav_s20\work\oc_val1.pkl` | 839492 | 2026-11-16 04:39 | .pkl | b3adb3318b740f196329b647ad76ee026cf136e733d2b109fc6668e41270dda0 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads\logs\H1_mav_s10.stdout` | 0 | 2026-11-17 09:56 | .stdout |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads\logs\H1c_mav_s10.stdout` | 0 | 2026-11-17 01:11 | .stdout |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads\logs\H1c_mav_s20.stdout` | 0 | 2026-11-17 12:15 | .stdout |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads\logs\H2_mav_s10.stdout` | 0 | 2026-11-17 01:11 | .stdout |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads\logs\H2_mav_s20.stdout` | 0 | 2026-11-17 12:15 | .stdout |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads\logs\H3_mav_s10.stdout` | 0 | 2026-11-17 12:15 | .stdout |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads\logs\H4_antifs_s0.stdout` | 30 | 2026-11-17 04:33 | .stdout | 1efe90601cc1809eb849ad370800eab43929fbe982093620bab3353d84393132 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads\logs\H4_antifs_s10.stdout` | 30 | 2026-11-17 04:33 | .stdout | 1efe90601cc1809eb849ad370800eab43929fbe982093620bab3353d84393132 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads\logs\H4_antifs_s20.stdout` | 30 | 2026-11-17 04:33 | .stdout | 1efe90601cc1809eb849ad370800eab43929fbe982093620bab3353d84393132 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads\logs\H4_mav_s10.stdout` | 0 | 2026-11-17 01:11 | .stdout |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads\logs\H4_mav_s20.stdout` | 0 | 2026-11-17 12:15 | .stdout |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads\logs\H5_antifs_s0.stdout` | 30 | 2026-11-17 04:33 | .stdout | 3218b0881e098a5f25b6b0219b92d26c4bbcdfdf42b5fe379be0adf6673eff70 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads\logs\H5_antifs_s10.stdout` | 30 | 2026-11-17 04:33 | .stdout | 3218b0881e098a5f25b6b0219b92d26c4bbcdfdf42b5fe379be0adf6673eff70 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads\logs\H5_antifs_s20.stdout` | 30 | 2026-11-17 04:33 | .stdout | 3218b0881e098a5f25b6b0219b92d26c4bbcdfdf42b5fe379be0adf6673eff70 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads\logs\H5_mav_s10.stdout` | 0 | 2026-11-17 01:11 | .stdout |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads\logs\H5_mav_s20.stdout` | 0 | 2026-11-17 12:15 | .stdout |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads\logs\H6_antifs_s0.stdout` | 30 | 2026-11-17 04:33 | .stdout | 2c570fce931eed8504ec90d8460308caeac2dfc5d16e6912773117a7d47e1cff |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads\logs\H6_antifs_s10.stdout` | 30 | 2026-11-17 04:33 | .stdout | 2c570fce931eed8504ec90d8460308caeac2dfc5d16e6912773117a7d47e1cff |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads\logs\H6_antifs_s20.stdout` | 30 | 2026-11-17 04:33 | .stdout | 2c570fce931eed8504ec90d8460308caeac2dfc5d16e6912773117a7d47e1cff |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads\logs\H6_mav_s10.stdout` | 0 | 2026-11-17 01:11 | .stdout |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads\logs\H6_mav_s20.stdout` | 0 | 2026-11-17 12:15 | .stdout |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/height\mav_htr\fe_deploy_fused_test_P.pkl` | 422800 | 2026-11-13 07:41 | .pkl | b7d21901788228eda1d3c8ebaae95139934aa6a056f91ceb13a94c4838c52361 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/height\mav_htr\fe_deploy_fused_val_P.pkl` | 205959 | 2026-11-13 07:41 | .pkl | 8bbc1bd7b4ccdb4a442dd0a664fecc3da7c9b2089373ed2f69a2509ce9857198 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/height\mav_rk\fe_deploy_fused_test_P.pkl` | 422800 | 2026-11-13 07:41 | .pkl | 12eb1577698861f469063b504dd8fa3211683b5fafe0eaf58733f559814cd385 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/height\mav_rk\fe_deploy_fused_val_P.pkl` | 205959 | 2026-11-13 07:41 | .pkl | fc333e13017e7f93c7920ce9a55a15fe398e784bd402e908f60776bdf40d11e5 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/hparam_1117/gamma/anti_s0/preds/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/hparam_1117/gamma/anti_s10/preds/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/hparam_1117/gamma/anti_s20/preds/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/hparam_1117/gamma/mav_s0/preds/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/hparam_1117/gamma/mav_s10/preds/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/hparam_1117/gamma/mav_s20/preds/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/hparam_1117/lam/mav_L050_s0/lib/__pycache__/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/hparam_1117/stu/g05/mav_s0/ou/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/hparam_1117/stu/g05/mav_s0/preds/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/hparam_1117\stu\g05\mav_s0\pseudo_tr1_g5.npz` | 5574350 | 2026-11-17 07:35 | .npz | 8768da0de02402d44f1b028b1e506122f5633de49284d63f21e141ca3fed88c6 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/hparam_1117\stu\g20\mav_s0\logs\ori_infer.log.try` | 0 | 2026-11-17 19:40 | .try |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/hparam_1117/stu/g20/mav_s0/ou/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/hparam_1117/stu/g20/mav_s0/preds/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/hparam_1117\stu\g20\mav_s0\pseudo_tr1_g20.npz` | 5574350 | 2026-11-17 16:51 | .npz | 6cafe393e35a68a7167fba645d2638568c69363acc074b960621662723c2f710 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/initabl\s0\kps_te1.pkl` | 209355 | 2026-11-14 16:53 | .pkl | 9062b0fa6cd45688fc11cbb441c9d81a0c88471d8a8c1d23ec3798ea4781c59c |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/initabl\s0\kps_tr1.pkl` | 616223 | 2026-11-14 16:53 | .pkl | 820b0bb2ef3a6bd21940616de219d39d735f8539c20994594de5033b50f359c8 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/initabl\s0\kps_val1.pkl` | 110298 | 2026-11-14 16:53 | .pkl | da1ce186f10b1123f4b2656f76675db6a2f538dd1c4d4fa5c801ab46cb4f478a |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/initabl/s0/lib/__pycache__/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/initabl/s0/ou/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/initabl/s0/preds/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/initabl/s0/recs/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/initabl/s0/ref/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/initabl\s10\kps_te1.pkl` | 209355 | 2026-11-14 17:35 | .pkl | 5643b8e63d5ee793f382b3231cb0ceedba28c7f1c93f7c1ae45c6c0fbac8fb93 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/initabl\s10\kps_tr1.pkl` | 616223 | 2026-11-14 17:35 | .pkl | aa90395c7519b78721e65d5e28119ee2e6d7619300be670aa1cb323ccaf1680e |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/initabl\s10\kps_val1.pkl` | 110298 | 2026-11-14 17:35 | .pkl | 56ee82b406127dba198fb20d4732cbf0c202c365b1a3f6a8377329b104d3f37e |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/initabl/s10/lib/__pycache__/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/initabl/s10/ou/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/initabl/s10/preds/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/initabl/s10/recs/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/initabl/s10/ref/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/initabl\s20\kps_te1.pkl` | 209355 | 2026-11-14 18:16 | .pkl | e26b0bd9cfb051fa807bccf680caea54c2c3e14993ed24586facc9639a87024e |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/initabl\s20\kps_tr1.pkl` | 616223 | 2026-11-14 18:16 | .pkl | 0345ce60d61d9c0333f549c2c424781ac57c683bf1ccb9fc586557dbfc0d9dea |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/initabl\s20\kps_val1.pkl` | 110298 | 2026-11-14 18:16 | .pkl | 7ddb73da9639923326959623c46384c53311075c35bbb55497d41413d5c7de16 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/initabl/s20/lib/__pycache__/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/initabl/s20/ou/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/initabl/s20/preds/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/initabl/s20/recs/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/initabl/s20/ref/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/k249\keys249.pkl` | 7957 | 2026-11-17 12:10 | .pkl | ec8975e8391a4a35e4d749e3f37e0aaedee77a7b4abf0aa5f91cafa0e5ea2733 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/k249/lib/__pycache__/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/lidraw\eval\T_a_s0.pth` | 111868502 | 2026-11-17 22:04 | .pth |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/lidraw\eval\T_b_s0.pth` | 111868502 | 2026-11-17 22:24 | .pth |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/lidraw\eval\pred_a_s0.npz` | 1095692 | 2026-11-17 22:19 | .npz | 3b477466744bcd1d917ad1c9470d36e5369b07fd40046af0de9edf54d2de91e2 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/lidraw\eval\read_te1_a_s0.pkl` | 9610110 | 2026-11-17 22:19 | .pkl | c378e238c5c7e8667e1e60edc707690453c6b21d3d7530887111dfd65af13cf5 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/lidraw\logs\ori_infer.log.try` | 0 | 2026-11-17 21:29 | .try |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/lidraw\logs\train_KZLs2.log` | 23549044 | 2026-11-17 21:27 | .log | c99da9a13a1dd82bdc18fd92fd7d789b17a4f289ed6576d6cf58416484ee9a7c |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/lidraw\logs\train_KZPs0.log` | 23123933 | 2026-11-15 01:26 | .log | 5907e7e14da778062085c99fc3f8c6bab225efd8e21def1376caf511e08c696e |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/lidraw\logs\train_KZPs1.log` | 18461226 | 2026-11-16 08:27 | .log | 9d360e19c56fbd1b292f192377c9b7f5d6a8223946b9c8ff327b052e061a12cd |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/lidraw\logs\train_KZPs2.log` | 23275079 | 2026-11-16 12:49 | .log | cb6999551d8c119e945d799c35f5c89ed86eb29c60b6c562cc24e65c8c685b86 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/lidraw\release\a_s2.pth` | 111858120 | 2026-11-17 21:29 | .pth |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/lidraw\release\b_s0.pth` | 111858120 | 2026-11-17 21:28 | .pth |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/lidraw\release\b_s1.pth` | 111858120 | 2026-11-17 21:28 | .pth |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/lidraw\release\b_s2.pth` | 111858120 | 2026-11-17 21:29 | .pth |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/lidraw/runs/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/lidraw/smoke/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/lowres/lib/__pycache__/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/orient_dist\lidar_share_rows.pkl` | 824097 | 2026-11-13 23:57 | .pkl | 3461f9e7b4bddf8619320687dacc286096567c51c6e30f571d68effb8d4c2e3e |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/ourdet_baseproto\check\ubzoom_contact.png` | 4636351 | 2026-11-17 01:24 | .png | c4239b1eb9dc55c7abacf14ec89d09968872c87a74a13b4a0873e3f291a809a9 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/ourdet_baseproto\logs\train_in2mav_pst.log` | 5152619 | 2026-11-16 16:34 | .log | 5a02200669a4b33d069c7f49e28212dd85bbbdd2e0b6f36c27bf2ee70779ef7c |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/ourdet_baseproto\logs\train_in2mav_pstg.log` | 4395339 | 2026-11-17 12:57 | .log | 51e5c52d155cc871c934b5acf57cc8e5511020f766bbe227f9e03aacb176df9e |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/ourdet_baseproto\logs\train_in2mav_pstgs.log` | 4478396 | 2026-11-17 15:34 | .log | 1210954d6d032cedd6501962d9f287b5d8a9cae5cbb6ff0575a955bd63238a7b |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/ourdet_baseproto\logs\train_in2mav_st.log` | 6282256 | 2026-11-16 18:26 | .log | ddcb206add9bb76d60e7fd4d9967915a836579fc356bb24b434057b00d59196a |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/ourdet_baseproto\logs\train_in2mav_ub.log` | 4065329 | 2026-11-16 17:52 | .log | da1efcf8c9c724755d3a04bfd3adf50c57a80d86b268142e409af02ac23d4749 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/ourdet_baseproto/preds/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/panelA\A11_size_sweep.pdf` | 26125 | 2026-11-14 03:03 | .pdf | cac2208e2039308c55661f7a3b1819b5a4300686d220c7fa643a870485e0b6da |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/panelA\A11_size_sweep.png` | 191540 | 2026-11-14 03:03 | .png | f8ba1ba715a3b1c8a41970696cc8b73429aacd8c6438da90c1621c5dd0c47dbf |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/panelA/cache/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/rgbst_equal\logs\gupnet.pid` | 6 | 2026-11-17 06:13 | .pid | 37a914408d00d3eb62373f0332befaa44a5815cea284c07cb916a3c460ecbbdb |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/rgbst_equal\logs\run.pid` | 6 | 2026-11-17 06:10 | .pid | a52e68095d40ace5e4dbad7e2e8c5fb3f9cc37a8731470e8520e6e3ac7a23967 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/rgbst_equal\logs\train_in2mav_naivest.log` | 5046221 | 2026-11-17 08:22 | .log | 05c1ec18c385018f80f0b9d3189e996d2c46cf8e7e9cca2b16588fc4e182fe85 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/rgbst_equal/preds/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/seeds\antifs_full_s10\kps_te1.pkl` | 70185 | 2026-11-14 02:44 | .pkl | f3d6ca6021c7a5c1109658e17dab6f241f062dd827c0dc9407c4a2d7ee87e46a |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/seeds\antifs_full_s10\kps_tr1.pkl` | 95245 | 2026-11-14 02:44 | .pkl | b23b93abfb778f31e198a061c78904b040859dfaa87630bb3a7b343a8bcfa192 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/seeds\antifs_full_s10\kps_val1.pkl` | 30842 | 2026-11-14 02:44 | .pkl | ed1621505d5c9d6eb5acb25e1f5768253a1ed53c135c9fa444d3c32100afcfcf |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/seeds/antifs_full_s10/lib/__pycache__/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/seeds/antifs_full_s10/ou/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/seeds/antifs_full_s10/preds/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/seeds/antifs_full_s10/recs/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/seeds/antifs_full_s10/ref/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/seeds\antifs_full_s10\wsim_cache.pkl` | 786084 | 2026-11-14 01:26 | .pkl | e0eac4e50df996dc500f6057c57ae77e8626f6a8c88c8fad8dcf798e15be357c |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/seeds\antifs_full_s10\wsim_read.pkl` | 1379854 | 2026-11-14 02:48 | .pkl | 3d5c1745c0eb26f10e8f8a14fa082c6d9837cd5620e547e28d49cf8047750f2c |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/seeds\antifs_full_s20\kps_te1.pkl` | 70185 | 2026-11-14 10:20 | .pkl | bddd8756f6c1074b01dc58df698a5029b5372ec1e35237c645fdfa1707c65a72 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/seeds\antifs_full_s20\kps_tr1.pkl` | 95245 | 2026-11-14 10:18 | .pkl | c1f1d99acf7429776d277559eb446911013cb69d84c34a8ff175d1853b0b4f36 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/seeds\antifs_full_s20\kps_val1.pkl` | 30842 | 2026-11-14 10:19 | .pkl | c2b8d9c0df9732d2aecffe2652cd7bb7a00d1bdf09c4a6d267471b372d61efdd |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/seeds/antifs_full_s20/lib/__pycache__/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/seeds/antifs_full_s20/ou/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/seeds/antifs_full_s20/preds/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/seeds/antifs_full_s20/recs/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/seeds/antifs_full_s20/ref/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/seeds\antifs_full_s20\wsim_cache.pkl` | 786084 | 2026-11-14 05:58 | .pkl | e0eac4e50df996dc500f6057c57ae77e8626f6a8c88c8fad8dcf798e15be357c |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/seeds\antifs_full_s20\wsim_read.pkl` | 1379854 | 2026-11-14 11:32 | .pkl | bd2512b2bcb8e1d8201f990ccc95945e2f43577ccd31f9c1efae8bae1532f6de |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/seeds\antifs_none_s10\kps_te1.pkl` | 70185 | 2026-11-14 04:58 | .pkl | d2239a478ba2a4ef74433bbcbe90dc3acf7235e9693ebef44f61cf20e802add7 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/seeds\antifs_none_s10\kps_tr1.pkl` | 95245 | 2026-11-14 04:58 | .pkl | 66f855fb8b46bc5d3a4ab1a01e8a4633a65fa69752927f47d783147b9253b706 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/seeds\antifs_none_s10\kps_val1.pkl` | 30842 | 2026-11-14 04:58 | .pkl | 8ea0ca407d4fa77c120b8775a3671d43acf7026b589521a33dfe0c73766bb44d |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/seeds/antifs_none_s10/lib/__pycache__/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/seeds/antifs_none_s10/ou/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/seeds/antifs_none_s10/preds/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/seeds/antifs_none_s10/recs/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/seeds/antifs_none_s10/ref/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/seeds\antifs_none_s10\wsim_cache.pkl` | 786084 | 2026-11-14 03:16 | .pkl | e0eac4e50df996dc500f6057c57ae77e8626f6a8c88c8fad8dcf798e15be357c |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/seeds\antifs_none_s10\wsim_read.pkl` | 1379854 | 2026-11-14 05:05 | .pkl | d9c12833b1487f14ca432b424e8a80316ffce5761336d15be93c984d3762b954 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/seeds\antifs_none_s20\kps_te1.pkl` | 70185 | 2026-11-14 15:09 | .pkl | 8ac3ea8c89f45a07b14c26fd5c1e6600bcdec0a79987c5c4695144356eaffe33 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/seeds\antifs_none_s20\kps_tr1.pkl` | 95245 | 2026-11-14 15:09 | .pkl | bd18eb9ae5cf883726c4608d8dfdd4b9e3c9c13971b6e5dc6e2a316bf7fb8ee5 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/seeds\antifs_none_s20\kps_val1.pkl` | 30842 | 2026-11-14 15:09 | .pkl | a8cd73c8750fa41af6099cc31961122ee320b6159bc299120d83990b5bb9782f |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/seeds/antifs_none_s20/lib/__pycache__/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/seeds/antifs_none_s20/ou/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/seeds/antifs_none_s20/preds/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/seeds/antifs_none_s20/recs/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/seeds/antifs_none_s20/ref/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/seeds\antifs_none_s20\wsim_cache.pkl` | 786084 | 2026-11-14 13:05 | .pkl | e0eac4e50df996dc500f6057c57ae77e8626f6a8c88c8fad8dcf798e15be357c |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/seeds\antifs_none_s20\wsim_read.pkl` | 1379854 | 2026-11-14 15:12 | .pkl | 17f8bb17879bd8eb33873e72b40aac27c784fee92a1cfaced8cfcfec5aaa352c |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/seeds\logs\antifs_full_s10.stdout` | 1034 | 2026-11-14 03:14 | .stdout | 893bad0d9865eb5e5abed987ff1873d60b5a91ddfe47f8069c2275da7994daab |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/seeds\logs\antifs_full_s20.stdout` | 1034 | 2026-11-14 13:05 | .stdout | b59faba0b31b09789f56783f1873d67fb91718b6712fbbeda135a9352aafad68 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/seeds\logs\antifs_none_s10.stdout` | 1044 | 2026-11-14 05:57 | .stdout | 5c84f2684acf3b0e3007909482cf04bfa5ca7c4ecaea0a1b914eab31b6d47820 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/seeds\logs\antifs_none_s20.stdout` | 1042 | 2026-11-14 15:41 | .stdout | b2870698d51fad389e83a059ef8ac0261573aed7dfa35da9e259b18ff7af38a6 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/seeds\logs\mav_full_s20.stdout` | 1023 | 2026-11-13 22:51 | .stdout | 923fefa08d8200c24aaca265a25b45fd97336ad3049f6c55571302b7c7c232ce |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/seeds\logs\mav_none_s10.stdout` | 1032 | 2026-11-13 20:04 | .stdout | 6fc285ecd81fd7f7f9690323f335459f9dc507319bb9d32a54fa21526c864cb4 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/seeds\logs\mav_none_s20.stdout` | 1032 | 2026-11-14 01:26 | .stdout | c9c89e29e1b235f76fd24797206a8dd73472e31036704c06ab354aaf46a13ec1 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/seeds\mav_full_s20\kps_te1.pkl` | 209355 | 2026-11-13 22:05 | .pkl | e26b0bd9cfb051fa807bccf680caea54c2c3e14993ed24586facc9639a87024e |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/seeds\mav_full_s20\kps_tr1.pkl` | 616223 | 2026-11-13 22:04 | .pkl | 0345ce60d61d9c0333f549c2c424781ac57c683bf1ccb9fc586557dbfc0d9dea |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/seeds\mav_full_s20\kps_val1.pkl` | 110298 | 2026-11-13 22:04 | .pkl | 7ddb73da9639923326959623c46384c53311075c35bbb55497d41413d5c7de16 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/seeds/mav_full_s20/lib/__pycache__/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/seeds/mav_full_s20/ou/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/seeds/mav_full_s20/preds/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/seeds/mav_full_s20/recs/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/seeds/mav_full_s20/ref/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/seeds\mav_full_s20\wsim_cache.pkl` | 1912979 | 2026-11-13 20:07 | .pkl | 3b27c15c20c112b28d385f9c6b7506d692c5a90850de16d51afc5ec24ada878c |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/seeds\mav_full_s20\wsim_read.pkl` | 10352776 | 2026-11-13 22:18 | .pkl | 6eb7b8a8a3edd61496b0c8fb6de1c8ee228da3a265d1b7d8e54305a94c27cbfb |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/seeds\mav_none_s10\kps_te1.pkl` | 209355 | 2026-11-13 19:21 | .pkl | 9de376d45e87652d77982f93542857a66b329209449da7ba7f05c129a8ff0a33 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/seeds\mav_none_s10\kps_tr1.pkl` | 616223 | 2026-11-13 19:21 | .pkl | b657f43ffe7fb2fa71aef20b1f56bd641b4643a11af064efad39a515535a6df5 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/seeds\mav_none_s10\kps_val1.pkl` | 110298 | 2026-11-13 19:21 | .pkl | 01073df2078acd2b4eba3b7a5952c19a392761f6b2bf41947cd250d4252aed7a |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/seeds/mav_none_s10/lib/__pycache__/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/seeds/mav_none_s10/ou/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/seeds/mav_none_s10/preds/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/seeds/mav_none_s10/recs/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/seeds/mav_none_s10/ref/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/seeds\mav_none_s10\wsim_cache.pkl` | 1912979 | 2026-11-13 17:41 | .pkl | 3b27c15c20c112b28d385f9c6b7506d692c5a90850de16d51afc5ec24ada878c |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/seeds\mav_none_s10\wsim_read.pkl` | 10352776 | 2026-11-13 19:33 | .pkl | 8288c9a333308de08c79102d37c495f9b5ab6d5ef1e15d429b8f2cbcd308272a |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/seeds\mav_none_s20\kps_te1.pkl` | 209355 | 2026-11-14 00:42 | .pkl | a699567babd87cfa744fda9d3cb4f3ff83a17083a751bcd810fda2bfc71f97d5 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/seeds\mav_none_s20\kps_tr1.pkl` | 616223 | 2026-11-14 00:42 | .pkl | 49c33f7d18acc246d9c7d1d6a1d39e65defcddacd5cca7638fc4040c76789d1b |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/seeds\mav_none_s20\kps_val1.pkl` | 110298 | 2026-11-14 00:42 | .pkl | c60af165da892a084c3289e2e940127d5fd3ee741e64bd7be6e19116485dbf90 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/seeds/mav_none_s20/lib/__pycache__/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/seeds/mav_none_s20/ou/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/seeds/mav_none_s20/preds/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/seeds/mav_none_s20/recs/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/seeds/mav_none_s20/ref/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/seeds\mav_none_s20\wsim_cache.pkl` | 1912979 | 2026-11-13 22:54 | .pkl | 3b27c15c20c112b28d385f9c6b7506d692c5a90850de16d51afc5ec24ada878c |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/seeds\mav_none_s20\wsim_read.pkl` | 10352776 | 2026-11-14 00:55 | .pkl | 41131e887aba9deccd84752a4aa016bc44db7817ffdc97fc90aa696b93db876d |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/senses\agree_rows.pkl` | 2069713 | 2026-11-13 04:04 | .pkl | 645b978fb63ad49d52064abf88ee7f1c4e4fb2bb6e5a13636cf1146aeab0b294 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree\E1\antifs_s0\kps_te1.pkl` | 70185 | 2026-11-15 04:16 | .pkl | 70a07052f006edeac20da1c5226c2b6409d9d6d792b82e8f4c5c9e44ea916aea |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree\E1\antifs_s0\kps_tr1.pkl` | 95245 | 2026-11-15 04:16 | .pkl | 47dac5d60de2119713ca51f3f017f6fefed8ff7977c8d9d3d5f525d483e0fb2c |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree\E1\antifs_s0\kps_val1.pkl` | 30842 | 2026-11-15 04:16 | .pkl | 9fbbf571ffcac76fb085a59d3c47668661388f12e6156e13febef530370273cb |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree/E1/antifs_s0/ou/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree/E1/antifs_s0/preds/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree/E1/antifs_s0/recs/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree/E1/antifs_s0/ref/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree\E1\antifs_s0\wsim_cache.pkl` | 786084 | 2026-11-15 02:40 | .pkl | e0eac4e50df996dc500f6057c57ae77e8626f6a8c88c8fad8dcf798e15be357c |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree\E1\antifs_s0\wsim_read.pkl` | 1379862 | 2026-11-15 04:20 | .pkl | 723332e3eee02b984a1d4ebbc7c757b71a48c0c820883b885c8349ce4c4004fb |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree\E1\antifs_s10\kps_te1.pkl` | 70185 | 2026-11-15 06:27 | .pkl | c03b02b573a9c98d8150d65bcf88c28ba29a11877edaa44ec49c35bbacbbb9de |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree\E1\antifs_s10\kps_tr1.pkl` | 95245 | 2026-11-15 06:27 | .pkl | 8ac779ebf925d09c67ce2f4b07ca4d5be62acb12279c3fcc2606a26ca5d65e3c |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree\E1\antifs_s10\kps_val1.pkl` | 30842 | 2026-11-15 06:27 | .pkl | b70e8c8da7c59b204e407bf9cb645cb228d88b4f9486760f01da20b4812a33f6 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree/E1/antifs_s10/ou/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree/E1/antifs_s10/preds/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree/E1/antifs_s10/recs/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree/E1/antifs_s10/ref/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree\E1\antifs_s10\wsim_cache.pkl` | 786084 | 2026-11-15 04:51 | .pkl | e0eac4e50df996dc500f6057c57ae77e8626f6a8c88c8fad8dcf798e15be357c |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree\E1\antifs_s10\wsim_read.pkl` | 1379866 | 2026-11-15 06:31 | .pkl | dc67a9eb3b73a015cdfa32509a1a269b925b8e3e99147b78eb989ebe1656a86a |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree\E1\antifs_s20\kps_te1.pkl` | 70185 | 2026-11-15 09:57 | .pkl | 4e985ce5086c9288fb15b50af4dff1585d16442c17971a951abeddc2c4b13ffd |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree\E1\antifs_s20\kps_tr1.pkl` | 95245 | 2026-11-15 09:57 | .pkl | f9cb203ec1e648467fe61ec2267701d4adde85ae5dc90d8e5f73a99f1409878a |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree\E1\antifs_s20\kps_val1.pkl` | 30842 | 2026-11-15 09:57 | .pkl | 378e2d4a27e6661ad6c87f07ca8e00c173edd4299716a95a2c4492a7310f292b |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree/E1/antifs_s20/ou/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree/E1/antifs_s20/preds/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree/E1/antifs_s20/recs/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree/E1/antifs_s20/ref/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree\E1\antifs_s20\wsim_cache.pkl` | 786084 | 2026-11-15 06:55 | .pkl | e0eac4e50df996dc500f6057c57ae77e8626f6a8c88c8fad8dcf798e15be357c |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree\E1\antifs_s20\wsim_read.pkl` | 1379866 | 2026-11-15 10:15 | .pkl | 4e94a3b572437e16fc33ec5a172f9d718637be43baff7279ce6cd24b255faa6b |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree\E1\mav_s0\kps_te1.pkl` | 209355 | 2026-11-15 05:39 | .pkl | e58bdeb3b18fc7cedcc5dbd627ae17718bdc3bbf487775eea0cf2812147b1301 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree\E1\mav_s0\kps_tr1.pkl` | 616223 | 2026-11-15 05:39 | .pkl | 97a992acbedd2fa87aea0c6cf1773ff83e9e3ad9cde90c1fa0c06fdd521d0932 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree\E1\mav_s0\kps_val1.pkl` | 110298 | 2026-11-15 05:39 | .pkl | 82df6b957d4f9820c75232bf02e178dd05233600290e68d4d73fb11fece8c094 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree/E1/mav_s0/ou/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree/E1/mav_s0/preds/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree/E1/mav_s0/recs/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree/E1/mav_s0/ref/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree\E1\mav_s0\wsim_cache.pkl` | 1912979 | 2026-11-15 02:44 | .pkl | 3b27c15c20c112b28d385f9c6b7506d692c5a90850de16d51afc5ec24ada878c |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree\E1\mav_s0\wsim_read.pkl` | 10352784 | 2026-11-15 05:59 | .pkl | d1bbe5cf1e0fed2c165874b6a15d17aec730baab7e751bdbd306f5e9db1b382b |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree\E1\mav_s10\kps_te1.pkl` | 209355 | 2026-11-15 11:46 | .pkl | 20a007ca1fdd4c52e38d40183b267d249852db62713035410db644ec43070781 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree\E1\mav_s10\kps_tr1.pkl` | 616223 | 2026-11-15 11:46 | .pkl | b0ac57f71c3c4fb7dc6cfdc0d1ad1825640ecef424b43baa64fd14fdc5425c94 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree\E1\mav_s10\kps_val1.pkl` | 110298 | 2026-11-15 11:46 | .pkl | b56b6cad1d8fef627cebfd96137b7bfb052ee1587ebb5ca015ccc786e76b699c |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree/E1/mav_s10/ou/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree/E1/mav_s10/preds/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree/E1/mav_s10/recs/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree/E1/mav_s10/ref/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree\E1\mav_s10\wsim_cache.pkl` | 1912979 | 2026-11-15 07:03 | .pkl | 3b27c15c20c112b28d385f9c6b7506d692c5a90850de16d51afc5ec24ada878c |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree\E1\mav_s10\wsim_read.pkl` | 10352788 | 2026-11-15 11:55 | .pkl | da0f912ab1b4fca1a96477035e149652bf324a1491bd85529530776e8cf1056e |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree\E1\mav_s20\kps_te1.pkl` | 209355 | 2026-11-15 17:20 | .pkl | c75914807f7379a0b7bab10c59d1ec55ea94dd2309ca2f5367b18176c6612e59 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree\E1\mav_s20\kps_tr1.pkl` | 616223 | 2026-11-15 17:20 | .pkl | 2916721d3a9e4e0c6809811bda6f1aaac4fe3e28eb029095c1257deec163b2fc |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree\E1\mav_s20\kps_val1.pkl` | 110298 | 2026-11-15 17:20 | .pkl | fa6b2534f2a73b022f777104ec90e221f83ad1646f6c3aff901035880beaad97 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree/E1/mav_s20/ou/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree/E1/mav_s20/preds/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree/E1/mav_s20/recs/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree/E1/mav_s20/ref/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree\E1\mav_s20\wsim_cache.pkl` | 1912979 | 2026-11-15 12:29 | .pkl | 3b27c15c20c112b28d385f9c6b7506d692c5a90850de16d51afc5ec24ada878c |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree\E1\mav_s20\wsim_read.pkl` | 10352788 | 2026-11-15 17:33 | .pkl | bc5347cfda4390facd09e2bd18c4599287cad83b05d54e1572679b8a4b9b3bf7 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree\E1b\antifs_s0\kps_te1.pkl` | 70185 | 2026-11-15 09:50 | .pkl | 27280523c239ed865e6f4f34e0552895d128e7a7a95774338cbb00a7bf404378 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree\E1b\antifs_s0\kps_tr1.pkl` | 95245 | 2026-11-15 09:50 | .pkl | 92553c7b20f0e754db886822dae035b98a0ef384c3b1ef9cf3e408598249dda7 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree\E1b\antifs_s0\kps_val1.pkl` | 30842 | 2026-11-15 09:50 | .pkl | f5254cf86bb99b2feb7e1e3ca21c0430a876d197635c4df8de81fc54ff5e2e27 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree/E1b/antifs_s0/ou/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree/E1b/antifs_s0/preds/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree/E1b/antifs_s0/recs/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree/E1b/antifs_s0/ref/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree\E1b\antifs_s0\wsim_cache.pkl` | 786084 | 2026-11-15 06:48 | .pkl | e0eac4e50df996dc500f6057c57ae77e8626f6a8c88c8fad8dcf798e15be357c |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree\E1b\antifs_s0\wsim_read.pkl` | 1379866 | 2026-11-15 09:53 | .pkl | 94417b59af43f821885e9209747c274ae59b421b81900b336ad4df781775e674 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree\E1b\antifs_s10\kps_te1.pkl` | 70185 | 2026-11-15 16:30 | .pkl | 6fcb7c36af7307c6611d5091cdb496e627d741b81104f129b8109b0560e69197 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree\E1b\antifs_s10\kps_tr1.pkl` | 95245 | 2026-11-15 16:30 | .pkl | f0295033fa543483d5712bd0df4fb6d52692ca3b49cd7509bbb958cda6fbac0d |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree\E1b\antifs_s10\kps_val1.pkl` | 30842 | 2026-11-15 16:30 | .pkl | 52bd36879ebd03d01941a8a0784239ebae4ae53e24fa90143fbea9d8fb158139 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree/E1b/antifs_s10/ou/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree/E1b/antifs_s10/preds/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree/E1b/antifs_s10/recs/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree/E1b/antifs_s10/ref/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree\E1b\antifs_s10\wsim_cache.pkl` | 786084 | 2026-11-15 13:27 | .pkl | e0eac4e50df996dc500f6057c57ae77e8626f6a8c88c8fad8dcf798e15be357c |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree\E1b\antifs_s10\wsim_read.pkl` | 1379870 | 2026-11-15 16:33 | .pkl | 1f3e51dfe1d166ab4c8533b5ae37314e6b300483174c43743934bc01f40b2897 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree\E1b\antifs_s20\kps_te1.pkl` | 70185 | 2026-11-16 03:41 | .pkl | da53b491c46fe68644ce2f3f1a51c9229af638811507a9d6fe9ec301822edcef |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree\E1b\antifs_s20\kps_tr1.pkl` | 95245 | 2026-11-16 03:41 | .pkl | 6fe791cdbad71ba54ffd6caf00e266df74bb458db4126e14535413a96d544be1 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree\E1b\antifs_s20\kps_val1.pkl` | 30842 | 2026-11-16 03:41 | .pkl | d35214d9d22d2cc68039204a1354179565e4e084c901f5687a6a2bc1b0b09c17 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree/E1b/antifs_s20/ou/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree/E1b/antifs_s20/preds/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree/E1b/antifs_s20/recs/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree/E1b/antifs_s20/ref/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree\E1b\antifs_s20\wsim_cache.pkl` | 786084 | 2026-11-16 01:44 | .pkl | e0eac4e50df996dc500f6057c57ae77e8626f6a8c88c8fad8dcf798e15be357c |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree\E1b\antifs_s20\wsim_read.pkl` | 1379870 | 2026-11-16 03:44 | .pkl | 0b3de1fc8e88898612e2e95cafdb275868caa6c756fa697942652f820eb0052c |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree\E1b\mav_s0\kps_te1.pkl` | 209355 | 2026-11-15 11:35 | .pkl | 2289ceaf8b3a867cda23d0a152ad1282a180e7b41642a1f7cc33f21b5b69474f |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree\E1b\mav_s0\kps_tr1.pkl` | 616223 | 2026-11-15 11:35 | .pkl | cd31f1fd29e7ee601d90328e224d3f3aec9583484238f9814682cafa740a158a |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree\E1b\mav_s0\kps_val1.pkl` | 110298 | 2026-11-15 11:35 | .pkl | 823939eb15cb60e1b0b15e782123c53ce9fa668c967cb61ba7696b8365f21009 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree/E1b/mav_s0/ou/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree/E1b/mav_s0/preds/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree/E1b/mav_s0/recs/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree/E1b/mav_s0/ref/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree\E1b\mav_s0\wsim_cache.pkl` | 1912979 | 2026-11-15 06:51 | .pkl | 3b27c15c20c112b28d385f9c6b7506d692c5a90850de16d51afc5ec24ada878c |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree\E1b\mav_s0\wsim_read.pkl` | 10352788 | 2026-11-15 11:53 | .pkl | 7386ecb01d2d38e9a9e7d6046dbe3b8d7e83c25e702c72bc7ecc8748adb9d72e |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree\E1b\mav_s10\kps_te1.pkl` | 209355 | 2026-11-15 18:15 | .pkl | 5a9f7951230cd5c240a7132172a374502983f289a00d24ceacec0b1621cb7be9 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree\E1b\mav_s10\kps_tr1.pkl` | 616223 | 2026-11-15 18:15 | .pkl | 6f8f75ed9f97da10dd2ecb9760aa3fb69f5ff241f67fa291ed2158abd44a6dc6 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree\E1b\mav_s10\kps_val1.pkl` | 110298 | 2026-11-15 18:15 | .pkl | 46aa4076c5c3216b833daf187aa75da2791d6851e69b9347550e905058e7d94d |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree/E1b/mav_s10/ou/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree/E1b/mav_s10/preds/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree/E1b/mav_s10/recs/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree/E1b/mav_s10/ref/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree\E1b\mav_s10\wsim_cache.pkl` | 1912979 | 2026-11-15 13:32 | .pkl | 3b27c15c20c112b28d385f9c6b7506d692c5a90850de16d51afc5ec24ada878c |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree\E1b\mav_s10\wsim_read.pkl` | 10352792 | 2026-11-15 18:33 | .pkl | 8463292cef4879baf6a6b9c9521267ca12b498f6523fd3c8aa9997f4e2a51a46 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree\E1b\mav_s20\kps_te1.pkl` | 209355 | 2026-11-16 03:36 | .pkl | 730d4ceb99bdb16a8eb250eb59275f50579d6ecea55ba16ebe21b82ac0575351 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree\E1b\mav_s20\kps_tr1.pkl` | 616223 | 2026-11-16 03:36 | .pkl | ec0258737ec1aa1374e5795c250f3efedb793c02c243f197b8921073d6aff24e |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree\E1b\mav_s20\kps_val1.pkl` | 110298 | 2026-11-16 03:36 | .pkl | 0f57e5201330d96ef8ef7e1724721770d04b6ef4a505189c133cb37d2001bae8 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree/E1b/mav_s20/ou/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree/E1b/mav_s20/preds/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree/E1b/mav_s20/recs/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree/E1b/mav_s20/ref/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree\E1b\mav_s20\wsim_cache.pkl` | 1912979 | 2026-11-16 00:02 | .pkl | 3b27c15c20c112b28d385f9c6b7506d692c5a90850de16d51afc5ec24ada878c |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree\E1b\mav_s20\wsim_read.pkl` | 10352792 | 2026-11-16 04:01 | .pkl | a435153c82330d4370e543c74ecab56449e7028949a6e93f4cea779682fe2b29 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree\E1geo\mav_s0\kpsbox_stu_te1.pkl` | 209355 | 2026-11-15 07:07 | .pkl | d15736bc66b39451a527aac5bcd9d8007bf7fd398e0c3f02e116134fc4739583 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree\E1geo\mav_s0\kpsbox_stu_tr1.pkl` | 616223 | 2026-11-15 07:07 | .pkl | 7ef8e396b24fba09518d3c3573289acf08ce7bec9862b7e77de36c1c6ff13fb9 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree\E1geo\mav_s0\kpsbox_stu_val1.pkl` | 110298 | 2026-11-15 07:07 | .pkl | f6eba9198c41aec6db4acea44140da39079ea5c9311bcc160bf588c973c5c571 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree\E1geo\mav_s0\kpsbox_teach_tr1.pkl` | 616223 | 2026-11-15 07:06 | .pkl | e9d5f8b790a5c1a5468144dd337837b065728912eaa26d4ae0e08bf6e3af9064 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree/E1geo/mav_s0/preds/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree\E1geo\mav_s10\kpsbox_stu_te1.pkl` | 209355 | 2026-11-15 12:33 | .pkl | 161b4f9d482d5fe5b5e426b4f62dfdb61810adbc3d07f357c40bb36bd08dfc20 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree\E1geo\mav_s10\kpsbox_stu_tr1.pkl` | 616223 | 2026-11-15 12:32 | .pkl | 0926c4710a640e7ab7f6f42e38e30ac2811811c78ad8ab82eba0fe7d280ce036 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree\E1geo\mav_s10\kpsbox_stu_val1.pkl` | 110298 | 2026-11-15 12:32 | .pkl | e6eeeccf1c2c8a133a3274e63ce6cf4489afff932c2dec6f570f5363b7add42d |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree\E1geo\mav_s10\kpsbox_teach_tr1.pkl` | 616223 | 2026-11-15 12:32 | .pkl | 0d438f138b8a6a6beadd8daffb376be97883cd537435e2b446fe9c4c64f87f96 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree/E1geo/mav_s10/preds/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree\E1geo\mav_s20\kpsbox_stu_te1.pkl` | 209355 | 2026-11-15 18:46 | .pkl | de88d4720b7cc605432af34f4a610623e4b2d4c04091d38ee3dd0ee90ad5344d |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree\E1geo\mav_s20\kpsbox_stu_tr1.pkl` | 616223 | 2026-11-15 18:46 | .pkl | b966c043f14a2a76171dd00d29cc83c5271823bd3374c3a8f99ea938785ab5a3 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree\E1geo\mav_s20\kpsbox_stu_val1.pkl` | 110298 | 2026-11-15 18:46 | .pkl | 6c2ef7f03e99e20bb9ddd111f53fac5c1c58073fb85730c4b24058ce2938e39a |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree\E1geo\mav_s20\kpsbox_teach_tr1.pkl` | 616223 | 2026-11-15 18:45 | .pkl | d18cbcebf2b9732f1189a44c02a6980cc1f258ea57f2fc933840f0779ff8a71d |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree/E1geo/mav_s20/preds/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree\E2\antifs_s10\kps_te1.pkl` | 70185 | 2026-11-15 18:46 | .pkl | f3d6ca6021c7a5c1109658e17dab6f241f062dd827c0dc9407c4a2d7ee87e46a |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree\E2\antifs_s10\kps_tr1.pkl` | 95245 | 2026-11-15 18:46 | .pkl | b23b93abfb778f31e198a061c78904b040859dfaa87630bb3a7b343a8bcfa192 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree\E2\antifs_s10\kps_val1.pkl` | 30842 | 2026-11-15 18:46 | .pkl | ed1621505d5c9d6eb5acb25e1f5768253a1ed53c135c9fa444d3c32100afcfcf |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree/E2/antifs_s10/ou/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree/E2/antifs_s10/preds/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree/E2/antifs_s10/recs/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree/E2/antifs_s10/ref/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree\E2\antifs_s10\work\oc_te1.pkl` | 889244 | 2026-11-15 18:47 | .pkl | 46c2dc25f1d50f0214c32bca78b379f9cb099683ebbb45edf98670c8d943bb7a |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree\E2\antifs_s10\work\oc_tr1.pkl` | 1123068 | 2026-11-15 18:47 | .pkl | f84b2fd5f6b18d10b7a1bceda0970a6b84d8451cd6fd0da143c9230dd62355e3 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree\E2\antifs_s10\work\oc_val1.pkl` | 389326 | 2026-11-15 18:47 | .pkl | 11a9e2cbeced9f03722a887f6564604e42a18d8e3434afbab1b609d039867dae |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree\E2\antifs_s20\kps_te1.pkl` | 70185 | 2026-11-15 20:55 | .pkl | bddd8756f6c1074b01dc58df698a5029b5372ec1e35237c645fdfa1707c65a72 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree\E2\antifs_s20\kps_tr1.pkl` | 95245 | 2026-11-15 20:55 | .pkl | c1f1d99acf7429776d277559eb446911013cb69d84c34a8ff175d1853b0b4f36 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree\E2\antifs_s20\kps_val1.pkl` | 30842 | 2026-11-15 20:55 | .pkl | c2b8d9c0df9732d2aecffe2652cd7bb7a00d1bdf09c4a6d267471b372d61efdd |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree/E2/antifs_s20/ou/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree/E2/antifs_s20/preds/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree/E2/antifs_s20/recs/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree/E2/antifs_s20/ref/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree\E2\antifs_s20\work\oc_te1.pkl` | 889244 | 2026-11-15 20:56 | .pkl | 823168b4388fc2265ae0fbeb6383345202ade587a6234af540e5537e9ae81d53 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree\E2\antifs_s20\work\oc_tr1.pkl` | 1123068 | 2026-11-15 20:56 | .pkl | 0a6b2032302f1e86c14127ecfd89283767b138c0c128911ab2eab0bbb3491898 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree\E2\antifs_s20\work\oc_val1.pkl` | 389326 | 2026-11-15 20:56 | .pkl | fee371245345cd9f8d1c0eb524247f8ec0b3f4c8b3a677870ee8a5751b05fcc4 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree\E2a\mav_s0\kps_te1.pkl` | 209355 | 2026-11-15 18:46 | .pkl | 9062b0fa6cd45688fc11cbb441c9d81a0c88471d8a8c1d23ec3798ea4781c59c |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree\E2a\mav_s0\kps_tr1.pkl` | 616223 | 2026-11-15 18:46 | .pkl | 820b0bb2ef3a6bd21940616de219d39d735f8539c20994594de5033b50f359c8 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree\E2a\mav_s0\kps_val1.pkl` | 110298 | 2026-11-15 18:46 | .pkl | da1ce186f10b1123f4b2656f76675db6a2f538dd1c4d4fa5c801ab46cb4f478a |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree/E2a/mav_s0/ou/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree/E2a/mav_s0/preds/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree/E2a/mav_s0/recs/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree/E2a/mav_s0/ref/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree\E2a\mav_s0\work\oc_te1.pkl` | 1707425 | 2026-11-15 19:02 | .pkl | bf3e21e180d2eebc48a3d57a843983e32a64e9f786f7fee3d4f5af8eececb7ed |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree\E2a\mav_s0\work\oc_tr1.pkl` | 4631227 | 2026-11-15 19:01 | .pkl | 1f5fd36e5b2e1a811c5c2c4fa5a734a53f7d22b28065b9f8493439b80a5c32f8 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree\E2a\mav_s0\work\oc_val1.pkl` | 839492 | 2026-11-15 19:01 | .pkl | 4ed7912b9fdc5c8bdc072cc8e63f45d8e72b34ace7acd4276040b868477b4e64 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree\E2b\mav_s0\kps_te1.pkl` | 209355 | 2026-11-15 21:32 | .pkl | 9062b0fa6cd45688fc11cbb441c9d81a0c88471d8a8c1d23ec3798ea4781c59c |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree\E2b\mav_s0\kps_tr1.pkl` | 616223 | 2026-11-15 21:32 | .pkl | 820b0bb2ef3a6bd21940616de219d39d735f8539c20994594de5033b50f359c8 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree\E2b\mav_s0\kps_val1.pkl` | 110298 | 2026-11-15 21:32 | .pkl | da1ce186f10b1123f4b2656f76675db6a2f538dd1c4d4fa5c801ab46cb4f478a |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree/E2b/mav_s0/ou/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree/E2b/mav_s0/preds/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree/E2b/mav_s0/recs/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree/E2b/mav_s0/ref/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree\E2b\mav_s0\work\oc_te1.pkl` | 1707425 | 2026-11-15 21:37 | .pkl | bf3e21e180d2eebc48a3d57a843983e32a64e9f786f7fee3d4f5af8eececb7ed |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree\E2b\mav_s0\work\oc_tr1.pkl` | 4631227 | 2026-11-15 21:37 | .pkl | 1f5fd36e5b2e1a811c5c2c4fa5a734a53f7d22b28065b9f8493439b80a5c32f8 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree\E2b\mav_s0\work\oc_val1.pkl` | 839492 | 2026-11-15 21:37 | .pkl | 4ed7912b9fdc5c8bdc072cc8e63f45d8e72b34ace7acd4276040b868477b4e64 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree\E4\antifs_s0\kps_te1.pkl` | 70185 | 2026-11-16 00:00 | .pkl | c724e99f4bc99fb433cbee964189516f815c048bf2e0bf72b017b531e62b290f |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree\E4\antifs_s0\kps_tr1.pkl` | 95245 | 2026-11-16 00:00 | .pkl | 8345e89169f7e6b26e94e5e80b33401887827a405af8b411e0e356c65f59516e |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree\E4\antifs_s0\kps_val1.pkl` | 30842 | 2026-11-16 00:00 | .pkl | 3092ad9ec4b3ff76b3f075354b77ab94ade27e9b72df04e0494947025255f38e |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree/E4/antifs_s0/ou/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree/E4/antifs_s0/preds/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree/E4/antifs_s0/recs/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree/E4/antifs_s0/ref/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree\E4\antifs_s0\wsim_cache.pkl` | 786084 | 2026-11-16 00:00 | .pkl | e0eac4e50df996dc500f6057c57ae77e8626f6a8c88c8fad8dcf798e15be357c |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree\E4\antifs_s0\wsim_read.pkl` | 1379862 | 2026-11-16 01:17 | .pkl | e082fdd98083ee110952d955a92019f0c9eaa0af28adc97ed461b6863f4de261 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree\E4\antifs_s10\kps_te1.pkl` | 70185 | 2026-11-17 09:33 | .pkl | d2239a478ba2a4ef74433bbcbe90dc3acf7235e9693ebef44f61cf20e802add7 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree\E4\antifs_s10\kps_tr1.pkl` | 95245 | 2026-11-17 09:33 | .pkl | 66f855fb8b46bc5d3a4ab1a01e8a4633a65fa69752927f47d783147b9253b706 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree\E4\antifs_s10\kps_val1.pkl` | 30842 | 2026-11-17 09:33 | .pkl | 8ea0ca407d4fa77c120b8775a3671d43acf7026b589521a33dfe0c73766bb44d |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree/E4/antifs_s10/ou/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree/E4/antifs_s10/preds/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree/E4/antifs_s10/recs/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree/E4/antifs_s10/ref/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree\E4\antifs_s10\wsim_cache.pkl` | 786084 | 2026-11-17 09:33 | .pkl | e0eac4e50df996dc500f6057c57ae77e8626f6a8c88c8fad8dcf798e15be357c |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree\E4\antifs_s10\wsim_read.pkl` | 1379866 | 2026-11-17 11:05 | .pkl | 6fafb884de79734136d05a5e77ec26362555aff030dce2f8f09ebdd8f0815bc4 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree\E4\antifs_s20\kps_te1.pkl` | 70185 | 2026-11-17 10:22 | .pkl | 8ac3ea8c89f45a07b14c26fd5c1e6600bcdec0a79987c5c4695144356eaffe33 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree\E4\antifs_s20\kps_tr1.pkl` | 95245 | 2026-11-17 10:22 | .pkl | bd18eb9ae5cf883726c4608d8dfdd4b9e3c9c13971b6e5dc6e2a316bf7fb8ee5 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree\E4\antifs_s20\kps_val1.pkl` | 30842 | 2026-11-17 10:22 | .pkl | a8cd73c8750fa41af6099cc31961122ee320b6159bc299120d83990b5bb9782f |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree/E4/antifs_s20/ou/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree/E4/antifs_s20/preds/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree/E4/antifs_s20/recs/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree/E4/antifs_s20/ref/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree\E4\antifs_s20\wsim_cache.pkl` | 786084 | 2026-11-17 10:22 | .pkl | e0eac4e50df996dc500f6057c57ae77e8626f6a8c88c8fad8dcf798e15be357c |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree\E4\antifs_s20\wsim_read.pkl` | 1379866 | 2026-11-17 11:16 | .pkl | b9e1b426e408491ce6109cf626c14c2b98c1c236f55ca0818f9694893c4c8a50 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree\E4\mav_s0\kps_te1.pkl` | 209355 | 2026-11-15 21:58 | .pkl | 177c8e71afbe59ad78a84badb49431cf488fbf7e4cf8b9f3ad6fee9dc501037f |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree\E4\mav_s0\kps_tr1.pkl` | 616223 | 2026-11-15 21:58 | .pkl | 07569d822633ac508c5655c122a13add45d093750bb46e7f07c723ad869b2514 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree\E4\mav_s0\kps_val1.pkl` | 110298 | 2026-11-15 21:58 | .pkl | d984dd9252b7e430e26ab372eb387b048843037a561e7acddc3a999036445b24 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree/E4/mav_s0/ou/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree/E4/mav_s0/preds/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree/E4/mav_s0/recs/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree/E4/mav_s0/ref/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree\E4\mav_s0\wsim_cache.pkl` | 1912979 | 2026-11-15 21:58 | .pkl | 3b27c15c20c112b28d385f9c6b7506d692c5a90850de16d51afc5ec24ada878c |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree\E4\mav_s0\wsim_read.pkl` | 10352784 | 2026-11-15 22:58 | .pkl | 2b772715440f74763299b23eccfc0644ffbebd933812fcd65b45b51455b0667b |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree\E4\mav_s10\kps_te1.pkl` | 209355 | 2026-11-17 04:33 | .pkl | 9de376d45e87652d77982f93542857a66b329209449da7ba7f05c129a8ff0a33 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree\E4\mav_s10\kps_tr1.pkl` | 616223 | 2026-11-17 04:33 | .pkl | b657f43ffe7fb2fa71aef20b1f56bd641b4643a11af064efad39a515535a6df5 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree\E4\mav_s10\kps_val1.pkl` | 110298 | 2026-11-17 04:33 | .pkl | 01073df2078acd2b4eba3b7a5952c19a392761f6b2bf41947cd250d4252aed7a |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree/E4/mav_s10/ou/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree/E4/mav_s10/preds/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree/E4/mav_s10/recs/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree/E4/mav_s10/ref/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree\E4\mav_s10\wsim_cache.pkl` | 1912979 | 2026-11-17 04:33 | .pkl | 3b27c15c20c112b28d385f9c6b7506d692c5a90850de16d51afc5ec24ada878c |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree\E4\mav_s10\wsim_read.pkl` | 10352788 | 2026-11-17 05:46 | .pkl | 29c5cf7fb54008fbc5b20ef9ee4dacd6c39ab8a9e9aeec1ee73826af377d8dd1 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree\E4\mav_s20\kps_te1.pkl` | 209355 | 2026-11-17 09:34 | .pkl | a699567babd87cfa744fda9d3cb4f3ff83a17083a751bcd810fda2bfc71f97d5 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree\E4\mav_s20\kps_tr1.pkl` | 616223 | 2026-11-17 09:34 | .pkl | 49c33f7d18acc246d9c7d1d6a1d39e65defcddacd5cca7638fc4040c76789d1b |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree\E4\mav_s20\kps_val1.pkl` | 110298 | 2026-11-17 09:34 | .pkl | c60af165da892a084c3289e2e940127d5fd3ee741e64bd7be6e19116485dbf90 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree/E4/mav_s20/ou/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree/E4/mav_s20/preds/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree/E4/mav_s20/recs/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree/E4/mav_s20/ref/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree\E4\mav_s20\wsim_cache.pkl` | 1912979 | 2026-11-17 09:34 | .pkl | 3b27c15c20c112b28d385f9c6b7506d692c5a90850de16d51afc5ec24ada878c |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree\E4\mav_s20\wsim_read.pkl` | 10352788 | 2026-11-17 11:14 | .pkl | 9952503b0bdac9381d75ce2b0a24e0bc1c3de874335a5da275a1f2e3feb36183 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree\logs\E.done` | 0 | 2026-11-16 06:27 | .done |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree\logs\E1_antifs_s0.stdout` | 0 | 2026-11-15 02:37 | .stdout |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree\logs\E1_antifs_s10.stdout` | 0 | 2026-11-15 04:50 | .stdout |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree\logs\E1_antifs_s20.stdout` | 0 | 2026-11-15 06:54 | .stdout |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree\logs\E1_mav_s0.stdout` | 0 | 2026-11-15 02:36 | .stdout |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree\logs\E1_mav_s10.stdout` | 0 | 2026-11-15 06:56 | .stdout |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree\logs\E1_mav_s20.stdout` | 0 | 2026-11-15 12:25 | .stdout |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree\logs\E1b_antifs_s0.stdout` | 0 | 2026-11-15 10:42 | .stdout |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree\logs\E1b_antifs_s10.stdout` | 0 | 2026-11-15 13:26 | .stdout |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree\logs\E1b_antifs_s20.stdout` | 0 | 2026-11-16 01:43 | .stdout |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree\logs\E1b_mav_s0.stdout` | 0 | 2026-11-15 18:45 | .stdout |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree\logs\E1b_mav_s10.stdout` | 0 | 2026-11-15 13:26 | .stdout |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree\logs\E1b_mav_s20.stdout` | 0 | 2026-11-15 23:56 | .stdout |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree\logs\E2_antifs_s10.stdout` | 0 | 2026-11-15 18:45 | .stdout |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree\logs\E2_antifs_s20.stdout` | 0 | 2026-11-15 20:54 | .stdout |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree\logs\E2a_mav_s0.stdout` | 0 | 2026-11-15 18:45 | .stdout |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree\logs\E2b_mav_s0.stdout` | 0 | 2026-11-15 21:32 | .stdout |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree\logs\E4_antifs_s0.stdout` | 0 | 2026-11-15 23:58 | .stdout |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree\logs\E4_antifs_s10.stdout` | 0 | 2026-11-17 11:38 | .stdout |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree\logs\E4_antifs_s20.stdout` | 0 | 2026-11-17 11:40 | .stdout |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree\logs\E4_mav_s0.stdout` | 0 | 2026-11-15 21:55 | .stdout |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree\logs\E4_mav_s10.stdout` | 0 | 2026-11-17 04:33 | .stdout |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree\logs\E4_mav_s20.stdout` | 0 | 2026-11-17 12:12 | .stdout |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree\logs\postA.done` | 0 | 2026-11-16 04:37 | .done |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree\logs\postB.done` | 0 | 2026-11-16 04:06 | .done |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree\logs\prioP1.done` | 0 | 2026-11-15 18:45 | .done |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree\logs\prioP2.done` | 0 | 2026-11-15 10:42 | .done |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree\logs\prio_post.stdout` | 318 | 2026-11-16 06:27 | .stdout | 80e5e784df904be33dbe15629fb8f0773fbf7be691c97ad5e6922da9ef4a6058 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree\plumb\ou_smoke\SMK_s0_smoke\ckpt_last.pth` | 111875403 | 2026-11-15 01:22 | .pth |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree\plumb\ref_E2a\sim_matte.npy` | 6553728 | 2026-11-15 01:19 | .npy | 9344e2382e4d06034b8c62cd721e61b5d7924488592ac1026a3495ede03bee7c |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree\plumb\ref_E2b\sim_matte.npy` | 6553728 | 2026-11-15 01:19 | .npy | 034c24c02eeb6bf05a734bd2146d785b92cc339360414ced09a6fd77d36af279 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/single_sbar\in2mav_single_flip_sbar_recs.pkl` | 1308978 | 2026-11-17 15:45 | .pkl | d9623503bfae5fcdf7b65c5530c250feafe832079182d462c2a5d7af0df77e9c |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/single_sbar\in2mav_single_flip_sbar_test.pkl` | 895975 | 2026-11-17 15:45 | .pkl | 729d89ae123c5e8bcbf29f8e04cefcd56dd739bd7b9ea0e8f9527d0eb278d65d |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/stage1seeds\logs\train_in_sd30_det0.log` | 3642571 | 2026-11-16 12:06 | .log | a19d1c31482725192e373afef3b2829b6fbbaf44f61639a7476831b285c5c0a4 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/stage1seeds\logs\train_in_sd30_det1.log` | 3555499 | 2026-11-16 12:40 | .log | e503d537ebe3794776252b7409fb40d81bd7c71170c74729ae5d05d60989e7db |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/stage1seeds\logs\train_in_sd30_det2.log` | 3566893 | 2026-11-16 13:19 | .log | 7198bbe86a11453660b03b93cee73a3361405005f227fd44f1fd79c4b8e3e714 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/stage1seeds\logs\train_in_sd30_irT0.log` | 23194498 | 2026-11-17 13:05 | .log | b18abe014b2fb4f648d7cf5836578ccd7197c8bf0b13a855e6b0cd5943bc4693 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/stage1seeds\logs\train_in_sd30_irlid0.log` | 5978157 | 2026-11-17 02:32 | .log | 0d8d946f9156affd4953f90abb3cb43fb2cfeb88dc62f2d8b7d8a8890a85be47 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/stage1seeds\logs\train_in_sd30_irlid1.log` | 23719244 | 2026-11-17 07:09 | .log | 5f4a60036347dd756c67cab7c372017dc258511a13ff83c1fecc4cb7abf68551 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/stage1seeds\logs\train_in_sd30_kdir0.log` | 24766934 | 2026-11-17 18:55 | .log | ec8587ad20aa965b3cc8f44c0edd1d7fe24865f76cb8a43d2716aa74fec7a9e0 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/stage1seeds\logs\train_in_sd30_kdir1.log.try` | 8164916 | 2026-11-17 18:55 | .try | 1e9f1d0c86b988d723a56fbfaeee50878ceda73eee34065cb095bc26d40f8e58 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/stage1seeds\logs\train_in_sd30_kps0.log` | 6270839 | 2026-11-17 09:10 | .log | c86c67f851e52dac3421324bf2ea0ef7c297a7f0a88a49e1f411e75d468b047c |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/stage1seeds\logs\train_in_sd30_lid0.log.try` | 12891995 | 2026-11-17 09:10 | .try | 75575e42acbba17099001212e6a1d291214e3a47af295ebcbdc6355f88b05704 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/stage1seeds\logs\train_in_sd30_rgb1.log` | 23117439 | 2026-11-17 05:56 | .log | 54912d33b761bf46851c88740c6fb49f5412aa3f01b38612acceedecff6dd891 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/stage1seeds\release\in_sd30\kzm_s30.pth` | 111873853 | 2026-11-17 02:32 | .pth |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/stage1seeds\release\in_sd30\kzm_s31.pth` | 111873853 | 2026-11-17 07:10 | .pth |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/stage1seeds\release\in_sd30\ramp_s30.pth` | 111817508 | 2026-11-16 12:06 | .pth |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/stage1seeds\release\in_sd30\ramp_s31.pth` | 111817508 | 2026-11-16 12:40 | .pth |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/stage1seeds\release\in_sd30\ramp_s32.pth` | 111817508 | 2026-11-16 13:19 | .pth |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/stage1seeds/runs/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/stage1seeds/shared/antifs_sd30/lib/__pycache__/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/stage1seeds/shared/antifs_sd40/lib/__pycache__/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/stage1seeds/shared/mav_sd30/lib/__pycache__/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/stage1seeds/shared/mav_sd30/preds/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/stage1seeds/shared/mav_sd30/recs/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/stage1seeds/shared/mav_sd30/ref/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/stage1seeds\shared\mav_sd30\wsim_cache.pkl` | 1918138 | 2026-11-17 17:28 | .pkl | 633ce653796d1afb3edb30f2429e71ce44084050ac83448fa85c1b7e61c9e8a9 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/stage1seeds/shared/mav_sd40/lib/__pycache__/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/stage1seeds\stage1\in_sd30\det0.done` | 0 | 2026-11-17 01:09 | .done |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/stage1seeds\stage1\in_sd30\det1.done` | 0 | 2026-11-17 01:09 | .done |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/stage1seeds\stage1\in_sd30\det2.done` | 0 | 2026-11-17 01:09 | .done |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/stage1seeds\stage1\in_sd30\irT0.done` | 0 | 2026-11-17 13:05 | .done |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/stage1seeds\stage1\in_sd30\irlid0.done` | 0 | 2026-11-17 02:32 | .done |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/stage1seeds\stage1\in_sd30\irlid1.done` | 0 | 2026-11-17 07:11 | .done |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/stage1seeds\stage1\in_sd30\kdir0.done` | 0 | 2026-11-17 18:55 | .done |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/stage1seeds\stage1\in_sd30\kps0.done` | 0 | 2026-11-17 09:10 | .done |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/stage1seeds\stage1\in_sd30\rgb0.done` | 0 | 2026-11-17 01:17 | .done |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/stage1seeds\stage1\in_sd30\rgb1.done` | 0 | 2026-11-17 05:56 | .done |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/strata_1117\feat_MAV6D.npz` | 347068 | 2026-11-17 21:28 | .npz | 25ba09eef3398f87d6a0adebb8937a810be5b2404de694477c9a595d0ba99ac1 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/strata_1117\feat_RealDrone3D.npz` | 148412 | 2026-11-17 21:19 | .npz | cbe0a00add7bf577b7cfe018fe87644cf0882cc7c75b5d041e56c73fc98e699f |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sup1/__pycache__/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sup1\adabn\bn\bn_s0.pth` | 111805423 | 2026-11-13 14:16 | .pth |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sup1\adabn\bn\bn_s1.pth` | 111805423 | 2026-11-13 14:22 | .pth |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sup1\adabn\bn\bn_s2.pth` | 111805423 | 2026-11-13 14:28 | .pth |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sup1/adabn/preds/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sup1/adabn/recs/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sup2\nogate\kps_te1.pkl` | 209355 | 2026-11-13 13:59 | .pkl | 9062b0fa6cd45688fc11cbb441c9d81a0c88471d8a8c1d23ec3798ea4781c59c |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sup2\nogate\kps_tr1.pkl` | 616223 | 2026-11-13 13:59 | .pkl | 820b0bb2ef3a6bd21940616de219d39d735f8539c20994594de5033b50f359c8 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sup2\nogate\kps_val1.pkl` | 110298 | 2026-11-13 13:59 | .pkl | da1ce186f10b1123f4b2656f76675db6a2f538dd1c4d4fa5c801ab46cb4f478a |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sup2/nogate/lib/__pycache__/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sup2/nogate/ou/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sup2/nogate/preds/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sup2/nogate/recs/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sup2/nogate/ref/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sup2\nogate\work\oc_te1.pkl` | 1707425 | 2026-11-13 13:59 | .pkl | bf3e21e180d2eebc48a3d57a843983e32a64e9f786f7fee3d4f5af8eececb7ed |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sup2\nogate\work\oc_tr1.pkl` | 4631227 | 2026-11-13 13:59 | .pkl | 1f5fd36e5b2e1a811c5c2c4fa5a734a53f7d22b28065b9f8493439b80a5c32f8 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sup2\nogate\work\oc_val1.pkl` | 839492 | 2026-11-13 13:59 | .pkl | 4ed7912b9fdc5c8bdc072cc8e63f45d8e72b34ace7acd4276040b868477b4e64 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sup2\nogate\work\pseudo_tr1.npz` | 5574350 | 2026-11-13 13:59 | .npz | 896e68a62b5d364efaae0ee4c77e396bbac6cfe1513238199695085432410934 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sup3/lib/__pycache__/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sup3\run\kps_te1.pkl` | 209355 | 2026-11-13 16:52 | .pkl | 5643b8e63d5ee793f382b3231cb0ceedba28c7f1c93f7c1ae45c6c0fbac8fb93 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sup3\run\kps_tr1.pkl` | 616223 | 2026-11-13 16:52 | .pkl | aa90395c7519b78721e65d5e28119ee2e6d7619300be670aa1cb323ccaf1680e |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sup3\run\kps_val1.pkl` | 110298 | 2026-11-13 16:52 | .pkl | 56ee82b406127dba198fb20d4732cbf0c202c365b1a3f6a8377329b104d3f37e |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sup3/run/ou/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sup3/run/preds/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sup3/run/recs/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sup3/run/ref/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sup3\run\wsim_cache.pkl` | 1912979 | 2026-11-13 14:34 | .pkl | 3b27c15c20c112b28d385f9c6b7506d692c5a90850de16d51afc5ec24ada878c |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sup3\run\wsim_read.pkl` | 10352736 | 2026-11-13 17:06 | .pkl | 716cfe01469f110a0dd1c9b6e05687b1ee481895ecdaa075ae7431ba36968e60 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/xdomain_1117\logs\A.stdout` | 608 | 2026-11-17 01:13 | .stdout | e52c756b9aa43bcc9bef9ff736ee345d5b96d8dd150c37038d1f76124ed5dfba |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/xdomain_1117\logs\B_run_x.stdout` | 992 | 2026-11-17 15:09 | .stdout | e5d7d0333ce67f5f6a04a4d6eaf2cc1f9e8f88fac49a63c64ff8afab29dc4805 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/xdomain_1117/preds/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/cmpfull/train_gupnet_s0.log` | 3084375 | 2026-10-30 06:53 | .log | 82c7b94544fa7d48383e8c682fb86e5c03bcd045fd6f9483e3e53e04a69f49d9 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/cmpfull/train_monocd_s0.log` | 6106570 | 2026-10-31 04:44 | .log | 5991110d1803e97f64b5ec446680c2cb209f940ab7b79a20427daceccae6e1dd |
| `E:/Open3DUAVDet/output/camnorm/geo1004/cmpfull/train_monodgp_s0.log` | 20804460 | 2026-10-31 11:55 | .log | 27a1026f7ddbbdc9e6f7ee2c1dac0ae1398d7723c772cc9e0807a7a5082cf878 |
| `E:/Open3DUAVDet/output/camnorm/geo1004/cmpfull/gupnet/.git/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/cmpfull/gupnet/code/lib/backbones/__pycache__/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/cmpfull/gupnet/code/lib/datasets/__pycache__/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/cmpfull/gupnet/code/lib/helpers/__pycache__/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/cmpfull/gupnet/code/lib/losses/__pycache__/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/cmpfull/gupnet/code/lib/models/__pycache__/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/cmpfull/gupnet/pdf/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/cmpfull/gupnet/picture/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/cmpfull/gupnet/smoke/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/cmpfull/monocd/.git/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/cmpfull/monocd/ImageSets/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/cmpfull/monocd/__pycache__/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/cmpfull/monocd/config/__pycache__/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/cmpfull/monocd/data/__pycache__/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/cmpfull/monocd/data/augmentations/__pycache__/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/cmpfull/monocd/data/datasets/__pycache__/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/cmpfull/monocd/data/samplers/__pycache__/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/cmpfull/monocd/data/transforms/__pycache__/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/cmpfull/monocd/engine/__pycache__/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/cmpfull/monocd/figures/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/cmpfull/monocd/model/__pycache__/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/cmpfull/monocd/model/backbone/DCNv2/__pycache__/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/cmpfull/monocd/model/backbone/__pycache__/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/cmpfull/monocd/model/head/__pycache__/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/cmpfull/monocd/model/layers/__pycache__/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/cmpfull/monocd/solver/__pycache__/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/cmpfull/monocd/structures/__pycache__/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/cmpfull/monocd/utils/__pycache__/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/cmpfull/monocd/wavedata/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/cmpfull/monodgp/.git/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/cmpfull/monodgp/figures/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/cmpfull/monodgp/lib/datasets/__pycache__/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/cmpfull/monodgp/lib/datasets/kitti/__pycache__/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/cmpfull/monodgp/lib/datasets/uav/__pycache__/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/cmpfull/monodgp/lib/helpers/__pycache__/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/cmpfull/monodgp/lib/losses/__pycache__/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/cmpfull/monodgp/lib/models/monodgp/__pycache__/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/cmpfull/monodgp/lib/models/monodgp/depth_predictor/__pycache__/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/cmpfull/monodgp/lib/models/monodgp/depth_predictor/ddn_loss/__pycache__/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/cmpfull/monodgp/lib/models/monodgp/ops/functions/__pycache__/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/cmpfull/monodgp/lib/models/monodgp/ops/modules/__pycache__/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/cmpfull/monodgp\lib\models\monodgp\ops\src\cuda\ms_deform_im2col_cuda.cuh` | 56020 | 2026-10-29 22:28 | .cuh | 1d70465589fb3eb70b7d842bcfe293ccc9b2dc85b73850c06902697b42571f7a |
| `E:/Open3DUAVDet/output/camnorm/geo1004/cmpfull/monodgp/smoke/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/cmpfull/monodgp/utils/__pycache__/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/cmpfull/monodetr\lib\models\monodetr\ops\src\cuda\ms_deform_im2col_cuda.cuh` | 54694 | 2026-10-31 00:35 | .cuh | 136729520587cbf0593209d692af61a1e8af5909bcf34b8bacb8844878496bad |
| `E:/Open3DUAVDet/output/camnorm/geo1004/cmpfull/monodetr\main_fig.png` | 1365093 | 2026-10-31 00:35 | .png | 6f86f6b10028dc3e0853f87ed265918b28463bfb15636e8977817148c8c52b6b |
| `E:/Open3DUAVDet/output/camnorm/geo1004/cmpfull/monodetr/smoke/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/cmpfull/monocop/assets/` | -1 |  | dir (not traversed) |  |
| `E:/Open3DUAVDet/output/camnorm/geo1004/cmpfull/monocop\lib\models\monocop\ops\src\cuda\ms_deform_im2col_cuda.cuh` | 54694 | 2026-10-31 00:35 | .cuh | 136729520587cbf0593209d692af61a1e8af5909bcf34b8bacb8844878496bad |
| `D:/UAV3D_baselines_pack/Open3DUAVDet/output/camnorm/geo1004/cmpfull/gupnet/pdf/` | -1 |  | dir (not traversed) |  |
| `D:/UAV3D_baselines_pack/Open3DUAVDet/output/camnorm/geo1004/cmpfull/gupnet/picture/` | -1 |  | dir (not traversed) |  |
| `D:/UAV3D_baselines_pack/Open3DUAVDet/output/camnorm/geo1004/cmpfull/monocd/ImageSets/` | -1 |  | dir (not traversed) |  |
| `D:/UAV3D_baselines_pack/Open3DUAVDet/output/camnorm/geo1004/cmpfull/monocd/figures/` | -1 |  | dir (not traversed) |  |
| `D:/UAV3D_baselines_pack/Open3DUAVDet/output/camnorm/geo1004/cmpfull/monocd/wavedata/` | -1 |  | dir (not traversed) |  |
| `D:/UAV3D_baselines_pack/Open3DUAVDet/output/camnorm/geo1004/cmpfull/monocop/assets/` | -1 |  | dir (not traversed) |  |
| `D:/UAV3D_baselines_pack/Open3DUAVDet\output\camnorm\geo1004\cmpfull\monodetr\main_fig.png` | 1365093 | 2026-10-31 00:35 | .png | 6f86f6b10028dc3e0853f87ed265918b28463bfb15636e8977817148c8c52b6b |
| `D:/UAV3D_baselines_pack/Open3DUAVDet/output/camnorm/geo1004/cmpfull/monodgp/figures/` | -1 |  | dir (not traversed) |  |
| `D:/UAV3D_baselines_pack/PCMD_code\output\camnorm\geo1004\kzst4\prereg_kzst4.json.sha256` | 66 | 2026-11-06 12:35 | .sha256 | 9a3d0db68f2f126d04b4cff626ceecde25aaa2137c75a47227f965a0301463d4 |
| `D:/UAV3D_baselines_pack/PCMD_code\output\camnorm\geo1004\kzst6\prereg_kzst6.json.sha256` | 66 | 2026-11-06 12:35 | .sha256 | 3f87c78de6365a486d57aa3176f590cc8443ae623006ce528eebb7f044e64044 |
| `D:/UAV3D_baselines_pack/PCMD_code\output\camnorm\geo1004\kzst7\prereg_kzst7.json.sha256` | 65 | 2026-11-06 12:35 | .sha256 | daae9416a856ef026b746d435be5f09d1e172aff75fcbbba67bc27eb3bf0cfd7 |
| `D:/UAV3D_baselines_pack/PCMD_code\output\camnorm\night1003\kzst2\prereg_stage2.json.sha256` | 86 | 2026-11-06 12:35 | .sha256 | 765a89d9ca56bf75fcf006a7dc08e0ffa657e207abec2e71dd0b85fa39d0166c |
| `D:/UAV3D_baselines_pack/PCMD_code\output\camnorm\night1003\scratch\e1sa\prereg.json.sha256` | 79 | 2026-11-06 12:35 | .sha256 | fef5890669f45ace249a59e4c99ebb2ec11ce8995be584ee90af4eb75cf1ef41 |
| `D:/UAV3D_baselines_pack/PCMD_code\output\camnorm\night1003\scratch\kzst1\prereg_stage1.json.sha256` | 86 | 2026-11-06 12:35 | .sha256 | 7d96b5eee97a07b6e13e636a758aaf49ea48a0e9ba55f82d64f08afeea0c92e0 |
| `D:/UAV3D_baselines_pack/PCMD_code\output\camnorm\night1003\scratch\repl\prereg_R.json.sha256` | 81 | 2026-11-06 12:35 | .sha256 | 5da01eb68aa9d510539d928a69487d3ef3a453748ba8f475b5bed52d968c5a39 |
| `D:/UAV3D_baselines_pack/data/gt/outdoor_test_meta.npz` | 713110 |  | GT meta (sealed test labels), not copied | 8ab681717153124a25d86c24abae74c41b3b54235df668d1fa6167f43ae6e017 |
| `D:/UAV3D_baselines_pack/data/gt/real_test_meta.npz` | 1864568 |  | GT meta (sealed test labels), not copied | 0bc3cdd174cf5b70f626f5a254947009f61dc867ac9a07027b2ee6ec3fea3f41 |
| `D:/UAV3D_baselines_pack/pretrained/gupnet_in_sim_s0/checkpoints/` | -1 |  | dir (not traversed) |  |
| `D:/UAV3D_baselines_pack/pretrained\monodgp_in_sim_s0\model_epoch_24.pth` | 169517610 | 2026-11-11 06:41 | .pth |  |
| `D:/UAV3D_baselines_pack/scripts/__pycache__/` | -1 |  | dir (not traversed) |  |
| `D:/UAV3D_baselines_pack/torch_home/hub/checkpoints/` | -1 |  | dir (not traversed) |  |
| `D:/UAV3D_pack_update_1112/data/gt/outdoor_test_meta.npz` | 672954 |  | GT meta (sealed test labels), not copied | dd05225e116f41ad8aa2ba0ff908ef1e4ccab03d48847cef85ceb08ff9132f45 |
| `D:/UAV3D_pack.zip` | 66948808032 |  | zip as shipped (sha256 below if <= 50 MB) |  |
| `D:/UAV3D_pack_fix_1113.zip` | 6154 |  | zip as shipped (sha256 below if <= 50 MB) | 09d4bdca45e7b2097216a69dd414034c5a054a4c4997184c1984bfc4d68c4518 |
| `D:/UAV3D_pack_fix_1113b.zip` | 6196 |  | zip as shipped (sha256 below if <= 50 MB) | af85ec404aa4699cbb773cff713d45025d54ab94c3130ea6f5c80e2e2a517ea9 |
| `D:/UAV3D_pack_fix_1114.zip` | 235893 |  | zip as shipped (sha256 below if <= 50 MB) | 7cd91af2202584355e5320272e6bc0dc7ebde5adf1325a7e744fdb5d7515b771 |
| `D:/UAV3D_pack_fix_1114b.zip` | 3049 |  | zip as shipped (sha256 below if <= 50 MB) | 7c9521e7643f4556de8e5b2636d8ee860073232fd853149250f1782aa289f5c4 |
| `D:/UAV3D_pack_fix_1115.zip` | 1929 |  | zip as shipped (sha256 below if <= 50 MB) | 16cb62bd15b1048a6201f6cb9616470879e6775310641331aaa408324fea1156 |
| `D:/UAV3D_pack_fix_1115b.zip` | 487411 |  | zip as shipped (sha256 below if <= 50 MB) | 2792a934404e46a04fe6a25180d57353911959e571554acf63c95e8490068dd0 |
| `D:/UAV3D_pack_fix_1116.zip` | 12629 |  | zip as shipped (sha256 below if <= 50 MB) | 0c13aa47ed404d221aab8b7113580b1d00058835d48d977d62de0607c553ed4d |
| `D:/UAV3D_pack_fix_1117.zip` | 129647 |  | zip as shipped (sha256 below if <= 50 MB) | c3dbd9b8b53736e1e97695e681c519703b85ff87ebb46abeeebcc3efa84cc0e2 |
| `D:/UAV3D_pack_fix_1117b.zip` | 99901 |  | zip as shipped (sha256 below if <= 50 MB) | 86a7b186f3f518369e40d00df50e81115f38846b4bb53cc8153b011e78a6cf5b |
| `D:/UAV3D_pack_fix_1117c.zip` | 100231 |  | zip as shipped (sha256 below if <= 50 MB) | 8684d7ccefb8732ab8856f431d0c46a11f183ae57ce528900b771716e15ab88a |
| `D:/UAV3D_pack_hotfix_1112b.zip` | 8983 |  | zip as shipped (sha256 below if <= 50 MB) | b031bd191a5db642beecffed75336dab40c86511bfe4fe773c087fa1bd8c59c7 |
| `D:/UAV3D_pack_hotfix_1112c.zip` | 10744 |  | zip as shipped (sha256 below if <= 50 MB) | cc40bdec5bfffc7f63031c7b4c7461a9ed2b10e2611715485441f8a361f0447b |
| `D:/UAV3D_pack_prio_1113.zip` | 3168 |  | zip as shipped (sha256 below if <= 50 MB) | eaebbfc068d59317526cdcf3116d3c3214bbce339b31ee568f28735cda9cc16c |
| `D:/UAV3D_pack_update_1112.zip` | 1363703525 |  | zip as shipped (sha256 below if <= 50 MB) |  |

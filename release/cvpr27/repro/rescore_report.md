# Rescore report (2026-11-17 23:16:23, 180 s)

Table cells: 328 checked, 326 reproduced exactly at the printed precision.
Main-text numbers and claims: 67 checked, 66 reproduced.

## Mismatches

| id | printed | recomputed | unrounded | how |
|---|---|---|---|---|
| tab2|PST w/ detection rate of row one|AP | 30.63 | 30.36 | 30.35962853941385 | derived: full pipeline, keep the 3349 most confident test frames (= frames detected by row one), others dropped |
| tab2|PST w/ detection rate of row one|UDS | 54.14 | 54.00 | 53.99935717438331 | derived: full pipeline, keep the 3349 most confident test frames (= frames detected by row one), others dropped |
| text|within 4.6 of real labels | 4.6 | 4.5 | 4.545445001675333 | UDS(real labels) - UDS(full), MAV6D |

## All checks

| id | printed | recomputed | ok |
|---|---|---|---|
| tab1|MonoDETR|mav|AP | 1.98 | 1.98 | yes |
| tab1|MonoDETR|mav|ATE | 1.091 | 1.091 | yes |
| tab1|MonoDETR|mav|ASE | 6.99 | 6.99 | yes |
| tab1|MonoDETR|mav|AOE | 20.23 | 20.23 | yes |
| tab1|MonoDETR|mav|mDR | 78.71 | 78.71 | yes |
| tab1|MonoDETR|mav|UDS | 22.67 | 22.67 | yes |
| tab1|MonoDETR|anti|AP | 0.00 | 0.00 | yes |
| tab1|MonoDETR|anti|ATE | 48.5 | 48.5 | yes |
| tab1|MonoDETR|anti|ASE | 16.8 | 16.8 | yes |
| tab1|MonoDETR|anti|AOE | 23.3 | 23.3 | yes |
| tab1|MonoDETR|anti|mDR | 64.10 | 64.10 | yes |
| tab1|MonoDETR|anti|UDS | 15.44 | 15.44 | yes |
| tab1|MonoDETR +ST|mav|AP | 1.89 | 1.89 | yes |
| tab1|MonoDETR +ST|mav|ATE | 0.991 | 0.991 | yes |
| tab1|MonoDETR +ST|mav|ASE | 3.90 | 3.90 | yes |
| tab1|MonoDETR +ST|mav|AOE | 18.54 | 18.54 | yes |
| tab1|MonoDETR +ST|mav|mDR | 68.81 | 68.81 | yes |
| tab1|MonoDETR +ST|mav|UDS | 25.74 | 25.74 | yes |
| tab1|MonoDETR +ST|anti|AP | 0.02 | 0.02 | yes |
| tab1|MonoDETR +ST|anti|ATE | 41.6 | 41.6 | yes |
| tab1|MonoDETR +ST|anti|ASE | 14.0 | 14.0 | yes |
| tab1|MonoDETR +ST|anti|AOE | 26.1 | 26.1 | yes |
| tab1|MonoDETR +ST|anti|mDR | 67.42 | 67.42 | yes |
| tab1|MonoDETR +ST|anti|UDS | 15.24 | 15.24 | yes |
| tab1|MonoDETR +PST|mav|AP | 40.97 | 40.97 | yes |
| tab1|MonoDETR +PST|mav|ATE | 0.304 | 0.304 | yes |
| tab1|MonoDETR +PST|mav|ASE | 2.51 | 2.51 | yes |
| tab1|MonoDETR +PST|mav|AOE | 16.52 | 16.52 | yes |
| tab1|MonoDETR +PST|mav|mDR | 96.14 | 96.14 | yes |
| tab1|MonoDETR +PST|mav|UDS | 59.45 | 59.45 | yes |
| tab1|MonoDETR +PST|anti|AP | 0.78 | 0.78 | yes |
| tab1|MonoDETR +PST|anti|ATE | 45.7 | 45.7 | yes |
| tab1|MonoDETR +PST|anti|ASE | 16.8 | 16.8 | yes |
| tab1|MonoDETR +PST|anti|AOE | 20.6 | 20.6 | yes |
| tab1|MonoDETR +PST|anti|mDR | 81.08 | 81.08 | yes |
| tab1|MonoDETR +PST|anti|UDS | 18.54 | 18.54 | yes |
| tab1|MonoCD|mav|AP | 3.68 | 3.68 | yes |
| tab1|MonoCD|mav|ATE | 1.013 | 1.013 | yes |
| tab1|MonoCD|mav|ASE | 8.58 | 8.58 | yes |
| tab1|MonoCD|mav|AOE | 23.83 | 23.83 | yes |
| tab1|MonoCD|mav|mDR | 41.64 | 41.64 | yes |
| tab1|MonoCD|mav|UDS | 18.55 | 18.55 | yes |
| tab1|MonoCD|anti|AP | 1.02 | 1.02 | yes |
| tab1|MonoCD|anti|ATE | 35.8 | 35.8 | yes |
| tab1|MonoCD|anti|ASE | 12.6 | 12.6 | yes |
| tab1|MonoCD|anti|AOE | 22.4 | 22.4 | yes |
| tab1|MonoCD|anti|mDR | 37.89 | 37.89 | yes |
| tab1|MonoCD|anti|UDS | 12.86 | 12.86 | yes |
| tab1|MonoCD +ST|mav|AP | 0.82 | 0.82 | yes |
| tab1|MonoCD +ST|mav|ATE | 0.787 | 0.787 | yes |
| tab1|MonoCD +ST|mav|ASE | 5.54 | 5.54 | yes |
| tab1|MonoCD +ST|mav|AOE | 20.29 | 20.29 | yes |
| tab1|MonoCD +ST|mav|mDR | 18.12 | 18.12 | yes |
| tab1|MonoCD +ST|mav|UDS | 18.99 | 18.99 | yes |
| tab1|MonoCD +ST|anti|AP | 0.29 | 0.29 | yes |
| tab1|MonoCD +ST|anti|ATE | 44.9 | 44.9 | yes |
| tab1|MonoCD +ST|anti|ASE | 17.8 | 17.8 | yes |
| tab1|MonoCD +ST|anti|AOE | 18.7 | 18.7 | yes |
| tab1|MonoCD +ST|anti|mDR | 22.80 | 22.80 | yes |
| tab1|MonoCD +ST|anti|UDS | 11.41 | 11.41 | yes |
| tab1|MonoCD +PST|mav|AP | 39.04 | 39.04 | yes |
| tab1|MonoCD +PST|mav|ATE | 0.309 | 0.309 | yes |
| tab1|MonoCD +PST|mav|ASE | 2.45 | 2.45 | yes |
| tab1|MonoCD +PST|mav|AOE | 18.12 | 18.12 | yes |
| tab1|MonoCD +PST|mav|mDR | 95.47 | 95.47 | yes |
| tab1|MonoCD +PST|mav|UDS | 58.08 | 58.08 | yes |
| tab1|MonoCD +PST|anti|AP | 13.66 | 13.66 | yes |
| tab1|MonoCD +PST|anti|ATE | 13.5 | 13.5 | yes |
| tab1|MonoCD +PST|anti|ASE | 14.1 | 14.1 | yes |
| tab1|MonoCD +PST|anti|AOE | 21.3 | 21.3 | yes |
| tab1|MonoCD +PST|anti|mDR | 81.33 | 81.33 | yes |
| tab1|MonoCD +PST|anti|UDS | 30.61 | 30.61 | yes |
| tab1|MonoDGP|mav|AP | 25.15 | 25.15 | yes |
| tab1|MonoDGP|mav|ATE | 0.544 | 0.544 | yes |
| tab1|MonoDGP|mav|ASE | 5.15 | 5.15 | yes |
| tab1|MonoDGP|mav|AOE | 16.55 | 16.55 | yes |
| tab1|MonoDGP|mav|mDR | 81.60 | 81.60 | yes |
| tab1|MonoDGP|mav|UDS | 43.43 | 43.43 | yes |
| tab1|MonoDGP|anti|AP | 2.76 | 2.76 | yes |
| tab1|MonoDGP|anti|ATE | 26.6 | 26.6 | yes |
| tab1|MonoDGP|anti|ASE | 15.9 | 15.9 | yes |
| tab1|MonoDGP|anti|AOE | 27.9 | 27.9 | yes |
| tab1|MonoDGP|anti|mDR | 88.36 | 88.36 | yes |
| tab1|MonoDGP|anti|UDS | 19.22 | 19.22 | yes |
| tab1|MonoDGP +ST|mav|AP | 25.01 | 25.01 | yes |
| tab1|MonoDGP +ST|mav|ATE | 0.463 | 0.463 | yes |
| tab1|MonoDGP +ST|mav|ASE | 3.79 | 3.79 | yes |
| tab1|MonoDGP +ST|mav|AOE | 15.25 | 15.25 | yes |
| tab1|MonoDGP +ST|mav|mDR | 83.06 | 83.06 | yes |
| tab1|MonoDGP +ST|mav|UDS | 46.53 | 46.53 | yes |
| tab1|MonoDGP +ST|anti|AP | 6.91 | 6.91 | yes |
| tab1|MonoDGP +ST|anti|ATE | 19.8 | 19.8 | yes |
| tab1|MonoDGP +ST|anti|ASE | 12.5 | 12.5 | yes |
| tab1|MonoDGP +ST|anti|AOE | 25.4 | 25.4 | yes |
| tab1|MonoDGP +ST|anti|mDR | 88.36 | 88.36 | yes |
| tab1|MonoDGP +ST|anti|UDS | 24.08 | 24.08 | yes |
| tab1|MonoDGP +PST|mav|AP | 42.68 | 42.68 | yes |
| tab1|MonoDGP +PST|mav|ATE | 0.318 | 0.318 | yes |
| tab1|MonoDGP +PST|mav|ASE | 2.50 | 2.50 | yes |
| tab1|MonoDGP +PST|mav|AOE | 16.67 | 16.67 | yes |
| tab1|MonoDGP +PST|mav|mDR | 99.47 | 99.47 | yes |
| tab1|MonoDGP +PST|mav|UDS | 60.55 | 60.55 | yes |
| tab1|MonoDGP +PST|anti|AP | 11.85 | 11.85 | yes |
| tab1|MonoDGP +PST|anti|ATE | 16.7 | 16.7 | yes |
| tab1|MonoDGP +PST|anti|ASE | 15.8 | 15.8 | yes |
| tab1|MonoDGP +PST|anti|AOE | 20.6 | 20.6 | yes |
| tab1|MonoDGP +PST|anti|mDR | 93.41 | 93.41 | yes |
| tab1|MonoDGP +PST|anti|UDS | 29.75 | 29.75 | yes |
| tab1|MonoCoP|mav|AP | 27.18 | 27.18 | yes |
| tab1|MonoCoP|mav|ATE | 0.514 | 0.514 | yes |
| tab1|MonoCoP|mav|ASE | 4.15 | 4.15 | yes |
| tab1|MonoCoP|mav|AOE | 15.46 | 15.46 | yes |
| tab1|MonoCoP|mav|mDR | 84.56 | 84.56 | yes |
| tab1|MonoCoP|mav|UDS | 46.68 | 46.68 | yes |
| tab1|MonoCoP|anti|AP | 1.49 | 1.49 | yes |
| tab1|MonoCoP|anti|ATE | 25.4 | 25.4 | yes |
| tab1|MonoCoP|anti|ASE | 12.5 | 12.5 | yes |
| tab1|MonoCoP|anti|AOE | 25.3 | 25.3 | yes |
| tab1|MonoCoP|anti|mDR | 91.39 | 91.39 | yes |
| tab1|MonoCoP|anti|UDS | 19.14 | 19.14 | yes |
| tab1|MonoCoP +ST|mav|AP | 26.79 | 26.79 | yes |
| tab1|MonoCoP +ST|mav|ATE | 0.438 | 0.438 | yes |
| tab1|MonoCoP +ST|mav|ASE | 3.47 | 3.47 | yes |
| tab1|MonoCoP +ST|mav|AOE | 14.51 | 14.51 | yes |
| tab1|MonoCoP +ST|mav|mDR | 86.40 | 86.40 | yes |
| tab1|MonoCoP +ST|mav|UDS | 48.71 | 48.71 | yes |
| tab1|MonoCoP +ST|anti|AP | 3.39 | 3.39 | yes |
| tab1|MonoCoP +ST|anti|ATE | 24.9 | 24.9 | yes |
| tab1|MonoCoP +ST|anti|ASE | 11.7 | 11.7 | yes |
| tab1|MonoCoP +ST|anti|AOE | 25.2 | 25.2 | yes |
| tab1|MonoCoP +ST|anti|mDR | 95.16 | 95.16 | yes |
| tab1|MonoCoP +ST|anti|UDS | 20.63 | 20.63 | yes |
| tab1|MonoCoP +PST|mav|AP | 44.60 | 44.60 | yes |
| tab1|MonoCoP +PST|mav|ATE | 0.276 | 0.276 | yes |
| tab1|MonoCoP +PST|mav|ASE | 2.60 | 2.60 | yes |
| tab1|MonoCoP +PST|mav|AOE | 16.79 | 16.79 | yes |
| tab1|MonoCoP +PST|mav|mDR | 99.71 | 99.71 | yes |
| tab1|MonoCoP +PST|mav|UDS | 61.90 | 61.90 | yes |
| tab1|MonoCoP +PST|anti|AP | 10.80 | 10.80 | yes |
| tab1|MonoCoP +PST|anti|ATE | 16.2 | 16.2 | yes |
| tab1|MonoCoP +PST|anti|ASE | 16.0 | 16.0 | yes |
| tab1|MonoCoP +PST|anti|AOE | 19.8 | 19.8 | yes |
| tab1|MonoCoP +PST|anti|mDR | 90.14 | 90.14 | yes |
| tab1|MonoCoP +PST|anti|UDS | 29.25 | 29.25 | yes |
| tab1|GUPNet|mav|AP | 6.05 | 6.05 | yes |
| tab1|GUPNet|mav|ATE | 1.418 | 1.418 | yes |
| tab1|GUPNet|mav|ASE | 14.64 | 14.64 | yes |
| tab1|GUPNet|mav|AOE | 23.61 | 23.61 | yes |
| tab1|GUPNet|mav|mDR | 65.34 | 65.34 | yes |
| tab1|GUPNet|mav|UDS | 18.54 | 18.54 | yes |
| tab1|GUPNet|anti|AP | 0.67 | 0.67 | yes |
| tab1|GUPNet|anti|ATE | 30.7 | 30.7 | yes |
| tab1|GUPNet|anti|ASE | 11.4 | 11.4 | yes |
| tab1|GUPNet|anti|AOE | 30.3 | 30.3 | yes |
| tab1|GUPNet|anti|mDR | 48.22 | 48.22 | yes |
| tab1|GUPNet|anti|UDS | 12.26 | 12.26 | yes |
| tab1|GUPNet +DANN|mav|AP | 10.47 | 10.47 | yes |
| tab1|GUPNet +DANN|mav|ATE | 1.316 | 1.316 | yes |
| tab1|GUPNet +DANN|mav|ASE | 13.46 | 13.46 | yes |
| tab1|GUPNet +DANN|mav|AOE | 21.88 | 21.88 | yes |
| tab1|GUPNet +DANN|mav|mDR | 72.71 | 72.71 | yes |
| tab1|GUPNet +DANN|mav|UDS | 22.05 | 22.05 | yes |
| tab1|GUPNet +DANN|anti|AP | 0.28 | 0.28 | yes |
| tab1|GUPNet +DANN|anti|ATE | 32.0 | 32.0 | yes |
| tab1|GUPNet +DANN|anti|ASE | 11.5 | 11.5 | yes |
| tab1|GUPNet +DANN|anti|AOE | 27.0 | 27.0 | yes |
| tab1|GUPNet +DANN|anti|mDR | 60.67 | 60.67 | yes |
| tab1|GUPNet +DANN|anti|UDS | 14.34 | 14.34 | yes |
| tab1|GUPNet +ST|mav|AP | 7.79 | 7.79 | yes |
| tab1|GUPNet +ST|mav|ATE | 0.861 | 0.861 | yes |
| tab1|GUPNet +ST|mav|ASE | 8.84 | 8.84 | yes |
| tab1|GUPNet +ST|mav|AOE | 19.44 | 19.44 | yes |
| tab1|GUPNet +ST|mav|mDR | 46.81 | 46.81 | yes |
| tab1|GUPNet +ST|mav|UDS | 21.20 | 21.20 | yes |
| tab1|GUPNet +ST|anti|AP | 0.02 | 0.02 | yes |
| tab1|GUPNet +ST|anti|ATE | 25.3 | 25.3 | yes |
| tab1|GUPNet +ST|anti|ASE | 11.3 | 11.3 | yes |
| tab1|GUPNet +ST|anti|AOE | 29.0 | 29.0 | yes |
| tab1|GUPNet +ST|anti|mDR | 27.90 | 27.90 | yes |
| tab1|GUPNet +ST|anti|UDS | 10.10 | 10.10 | yes |
| tab1|GUPNet +PST|mav|AP | 38.91 | 38.91 | yes |
| tab1|GUPNet +PST|mav|ATE | 0.321 | 0.321 | yes |
| tab1|GUPNet +PST|mav|ASE | 2.80 | 2.80 | yes |
| tab1|GUPNet +PST|mav|AOE | 16.92 | 16.92 | yes |
| tab1|GUPNet +PST|mav|mDR | 94.96 | 94.96 | yes |
| tab1|GUPNet +PST|mav|UDS | 57.61 | 57.61 | yes |
| tab1|GUPNet +PST|anti|AP | 15.97 | 15.97 | yes |
| tab1|GUPNet +PST|anti|ATE | 14.3 | 14.3 | yes |
| tab1|GUPNet +PST|anti|ASE | 15.7 | 15.7 | yes |
| tab1|GUPNet +PST|anti|AOE | 19.5 | 19.5 | yes |
| tab1|GUPNet +PST|anti|mDR | 93.03 | 93.03 | yes |
| tab1|GUPNet +PST|anti|UDS | 33.21 | 33.21 | yes |
| tab1|CenterNet|mav|AP | 13.09 | 13.09 | yes |
| tab1|CenterNet|mav|ATE | 0.666 | 0.666 | yes |
| tab1|CenterNet|mav|ASE | 5.92 | 5.92 | yes |
| tab1|CenterNet|mav|AOE | 20.42 | 20.42 | yes |
| tab1|CenterNet|mav|mDR | 69.79 | 69.79 | yes |
| tab1|CenterNet|mav|UDS | 32.59 | 32.59 | yes |
| tab1|CenterNet|anti|AP | 2.93 | 2.93 | yes |
| tab1|CenterNet|anti|ATE | 47.0 | 47.0 | yes |
| tab1|CenterNet|anti|ASE | 37.0 | 37.0 | yes |
| tab1|CenterNet|anti|AOE | 26.3 | 26.3 | yes |
| tab1|CenterNet|anti|mDR | 71.95 | 71.95 | yes |
| tab1|CenterNet|anti|UDS | 17.22 | 17.22 | yes |
| tab1|CenterNet +DANN|mav|AP | 7.33 | 7.33 | yes |
| tab1|CenterNet +DANN|mav|ATE | 0.894 | 0.894 | yes |
| tab1|CenterNet +DANN|mav|ASE | 6.79 | 6.79 | yes |
| tab1|CenterNet +DANN|mav|AOE | 23.75 | 23.75 | yes |
| tab1|CenterNet +DANN|mav|mDR | 80.60 | 80.60 | yes |
| tab1|CenterNet +DANN|mav|UDS | 26.39 | 26.39 | yes |
| tab1|CenterNet +DANN|anti|AP | 0.01 | 0.01 | yes |
| tab1|CenterNet +DANN|anti|ATE | 56.8 | 56.8 | yes |
| tab1|CenterNet +DANN|anti|ASE | 20.4 | 20.4 | yes |
| tab1|CenterNet +DANN|anti|AOE | 23.5 | 23.5 | yes |
| tab1|CenterNet +DANN|anti|mDR | 52.74 | 52.74 | yes |
| tab1|CenterNet +DANN|anti|UDS | 13.97 | 13.97 | yes |
| tab1|CenterNet +ST|mav|AP | 14.88 | 14.88 | yes |
| tab1|CenterNet +ST|mav|ATE | 0.564 | 0.564 | yes |
| tab1|CenterNet +ST|mav|ASE | 4.17 | 4.17 | yes |
| tab1|CenterNet +ST|mav|AOE | 20.16 | 20.16 | yes |
| tab1|CenterNet +ST|mav|mDR | 73.96 | 73.96 | yes |
| tab1|CenterNet +ST|mav|UDS | 37.52 | 37.52 | yes |
| tab1|CenterNet +ST|anti|AP | 5.47 | 5.47 | yes |
| tab1|CenterNet +ST|anti|ATE | 25.3 | 25.3 | yes |
| tab1|CenterNet +ST|anti|ASE | 26.2 | 26.2 | yes |
| tab1|CenterNet +ST|anti|AOE | 23.1 | 23.1 | yes |
| tab1|CenterNet +ST|anti|mDR | 59.68 | 59.68 | yes |
| tab1|CenterNet +ST|anti|UDS | 17.67 | 17.67 | yes |
| tab1|CenterNet +PST|mav|AP | 23.84 | 23.84 | yes |
| tab1|CenterNet +PST|mav|ATE | 0.485 | 0.485 | yes |
| tab1|CenterNet +PST|mav|ASE | 2.94 | 2.94 | yes |
| tab1|CenterNet +PST|mav|AOE | 20.89 | 20.89 | yes |
| tab1|CenterNet +PST|mav|mDR | 95.88 | 95.88 | yes |
| tab1|CenterNet +PST|mav|UDS | 47.10 | 47.10 | yes |
| tab1|CenterNet +PST|anti|AP | 14.77 | 14.77 | yes |
| tab1|CenterNet +PST|anti|ATE | 12.8 | 12.8 | yes |
| tab1|CenterNet +PST|anti|ASE | 15.3 | 15.3 | yes |
| tab1|CenterNet +PST|anti|AOE | 23.2 | 23.2 | yes |
| tab1|CenterNet +PST|anti|mDR | 72.83 | 72.83 | yes |
| tab1|CenterNet +PST|anti|UDS | 30.01 | 30.01 | yes |
| tab1|CenterNet$^*$|mav|AP | 26.87 | 26.87 | yes |
| tab1|CenterNet$^*$|mav|ATE | 0.283 | 0.283 | yes |
| tab1|CenterNet$^*$|mav|ASE | 1.74 | 1.74 | yes |
| tab1|CenterNet$^*$|mav|AOE | 20.32 | 20.32 | yes |
| tab1|CenterNet$^*$|mav|mDR | 69.77 | 69.77 | yes |
| tab1|CenterNet$^*$|mav|UDS | 49.51 | 49.51 | yes |
| tab1|CenterNet$^*$|anti|AP | 16.04 | 16.04 | yes |
| tab1|CenterNet$^*$|anti|ATE | 13.2 | 13.2 | yes |
| tab1|CenterNet$^*$|anti|ASE | 15.4 | 15.4 | yes |
| tab1|CenterNet$^*$|anti|AOE | 26.1 | 26.1 | yes |
| tab1|CenterNet$^*$|anti|mDR | 78.54 | 78.54 | yes |
| tab1|CenterNet$^*$|anti|UDS | 30.55 | 30.55 | yes |
| tab1|CenterNet$^*$, full pipeline of PST|mav|AP | 40.93 | 40.93 | yes |
| tab1|CenterNet$^*$, full pipeline of PST|mav|ATE | 0.306 | 0.306 | yes |
| tab1|CenterNet$^*$, full pipeline of PST|mav|ASE | 1.75 | 1.75 | yes |
| tab1|CenterNet$^*$, full pipeline of PST|mav|AOE | 14.48 | 14.48 | yes |
| tab1|CenterNet$^*$, full pipeline of PST|mav|mDR | 96.17 | 96.17 | yes |
| tab1|CenterNet$^*$, full pipeline of PST|mav|UDS | 60.81 | 60.81 | yes |
| tab1|CenterNet$^*$, full pipeline of PST|anti|AP | 23.28 | 23.28 | yes |
| tab1|CenterNet$^*$, full pipeline of PST|anti|ATE | 11.9 | 11.9 | yes |
| tab1|CenterNet$^*$, full pipeline of PST|anti|ASE | 16.2 | 16.2 | yes |
| tab1|CenterNet$^*$, full pipeline of PST|anti|AOE | 17.3 | 17.3 | yes |
| tab1|CenterNet$^*$, full pipeline of PST|anti|mDR | 94.32 | 94.32 | yes |
| tab1|CenterNet$^*$, full pipeline of PST|anti|UDS | 38.69 | 38.69 | yes |
| tab1|CenterNet, real labels|mav|AP | 45.46 | 45.46 | yes |
| tab1|CenterNet, real labels|mav|ATE | 0.276 | 0.276 | yes |
| tab1|CenterNet, real labels|mav|ASE | 0.05 | 0.05 | yes |
| tab1|CenterNet, real labels|mav|AOE | 17.62 | 17.62 | yes |
| tab1|CenterNet, real labels|mav|mDR | 99.94 | 99.94 | yes |
| tab1|CenterNet, real labels|mav|UDS | 65.36 | 65.36 | yes |
| tab1|CenterNet, real labels|anti|AP | 5.73 | 5.73 | yes |
| tab1|CenterNet, real labels|anti|ATE | 19.5 | 19.5 | yes |
| tab1|CenterNet, real labels|anti|ASE | 1.2 | 1.2 | yes |
| tab1|CenterNet, real labels|anti|AOE | 24.7 | 24.7 | yes |
| tab1|CenterNet, real labels|anti|mDR | 84.91 | 84.91 | yes |
| tab1|CenterNet, real labels|anti|UDS | 34.39 | 34.39 | yes |
| tab2|CenterNet$^*$|AP | 26.87 | 26.87 | yes |
| tab2|CenterNet$^*$|ATE | 0.283 | 0.283 | yes |
| tab2|CenterNet$^*$|ASE | 1.74 | 1.74 | yes |
| tab2|CenterNet$^*$|AOE | 20.32 | 20.32 | yes |
| tab2|CenterNet$^*$|mDR | 69.77 | 69.77 | yes |
| tab2|CenterNet$^*$|UDS | 49.51 | 49.51 | yes |
| tab2|+ Step 1, pseudo box|AP | 34.04 | 34.04 | yes |
| tab2|+ Step 1, pseudo box|ATE | 0.356 | 0.356 | yes |
| tab2|+ Step 1, pseudo box|ASE | 2.54 | 2.54 | yes |
| tab2|+ Step 1, pseudo box|AOE | 25.44 | 25.44 | yes |
| tab2|+ Step 1, pseudo box|mDR | 96.23 | 96.23 | yes |
| tab2|+ Step 1, pseudo box|UDS | 53.38 | 53.38 | yes |
| tab2|+ Step 2, pseudo orientation|AP | 34.04 | 34.04 | yes |
| tab2|+ Step 2, pseudo orientation|ATE | 0.355 | 0.355 | yes |
| tab2|+ Step 2, pseudo orientation|ASE | 2.54 | 2.54 | yes |
| tab2|+ Step 2, pseudo orientation|AOE | 14.48 | 14.48 | yes |
| tab2|+ Step 2, pseudo orientation|mDR | 96.21 | 96.21 | yes |
| tab2|+ Step 2, pseudo orientation|UDS | 55.78 | 55.78 | yes |
| tab2|+ Step 3, pseudo depth, height ratio and fusion|AP | 40.93 | 40.93 | yes |
| tab2|+ Step 3, pseudo depth, height ratio and fusion|ATE | 0.306 | 0.306 | yes |
| tab2|+ Step 3, pseudo depth, height ratio and fusion|ASE | 1.75 | 1.75 | yes |
| tab2|+ Step 3, pseudo depth, height ratio and fusion|AOE | 14.48 | 14.48 | yes |
| tab2|+ Step 3, pseudo depth, height ratio and fusion|mDR | 96.17 | 96.17 | yes |
| tab2|+ Step 3, pseudo depth, height ratio and fusion|UDS | 60.81 | 60.81 | yes |
| tab2|PST w/o composites in the depth refiner|AP | 31.97 | 31.97 | yes |
| tab2|PST w/o composites in the depth refiner|ATE | 0.375 | 0.375 | yes |
| tab2|PST w/o composites in the depth refiner|ASE | 1.66 | 1.66 | yes |
| tab2|PST w/o composites in the depth refiner|AOE | 15.19 | 15.19 | yes |
| tab2|PST w/o composites in the depth refiner|mDR | 95.31 | 95.31 | yes |
| tab2|PST w/o composites in the depth refiner|UDS | 55.32 | 55.32 | yes |
| tab2|PST w/ detection rate of row one|AP | 30.63 | 30.36 | **NO** |
| tab2|PST w/ detection rate of row one|ATE | 0.247 | 0.247 | yes |
| tab2|PST w/ detection rate of row one|ASE | 1.36 | 1.36 | yes |
| tab2|PST w/ detection rate of row one|AOE | 11.96 | 11.96 | yes |
| tab2|PST w/ detection rate of row one|mDR | 69.77 | 69.77 | yes |
| tab2|PST w/ detection rate of row one|UDS | 54.14 | 54.00 | **NO** |
| tab3|PST|mav|mean | 60.93 | 60.93 | yes |
| tab3|PST|mav|sd | 1.72 | 1.72 | yes |
| tab3|PST|anti|mean | 41.71 | 41.71 | yes |
| tab3|PST|anti|sd | 0.31 | 0.31 | yes |
| tab3|PST w/o sensors|mav|mean | 56.71 | 56.71 | yes |
| tab3|PST w/o sensors|mav|sd | 4.39 | 4.39 | yes |
| tab3|PST w/o sensors|anti|mean | 39.23 | 39.23 | yes |
| tab3|PST w/o sensors|anti|sd | 1.49 | 1.49 | yes |
| tab3|PST w/o sensors and composites|mav|mean | 58.00 | 58.00 | yes |
| tab3|PST w/o sensors and composites|mav|sd | 2.32 | 2.32 | yes |
| tab3|PST w/o sensors and composites|anti|mean | 35.11 | 35.11 | yes |
| tab3|PST w/o sensors and composites|anti|sd | 3.29 | 3.29 | yes |
| tab3|PST w/ RGB-only teachers|mav|mean | 58.08 | 58.08 | yes |
| tab3|PST w/ RGB-only teachers|mav|sd | 0.74 | 0.74 | yes |
| tab3|PST w/ RGB-only teachers|anti|mean | 34.83 | 34.83 | yes |
| tab3|PST w/ RGB-only teachers|anti|sd | 3.61 | 3.61 | yes |
| text|gain PST-sim MAV min | 14.5 | 14.5 | yes |
| text|gain PST-sim MAV max | 39.5 | 39.5 | yes |
| text|gain PST-sim RD min | 3.1 | 3.1 | yes |
| text|gain PST-sim RD max | 21.0 | 21.0 | yes |
| text|ATE sim 5 baselines min | 0.51 | 0.51 | yes |
| text|ATE sim 5 baselines max | 1.42 | 1.42 | yes |
| text|ATE PST 5 baselines min | 0.28 | 0.28 | yes |
| text|ATE PST 5 baselines max | 0.32 | 0.32 | yes |
| text|ASE of every detector below 3 cm (MAV, +PST) | ASE < 3 cm for all six +PST rows | max 2.94 | yes |
| text|min mDR +PST MAV | 94.96 | 94.96 | yes |
| text|RD: +PST lowers ATE and AOE for all six | ATE and AOE lower than sim-only for all six on RD | checked 12 pairs | yes |
| text|ST ATE MAV min | 0.44 | 0.44 | yes |
| text|ST ATE MAV max | 0.99 | 0.99 | yes |
| text|ST changes UDS by at most 5 | |UDS(+ST) - UDS(sim)| <= 5 for all 12 | max |change| 4.93 | yes |
| text|ST falls in N of 12 | 3 | 3 | yes |
| text|DANN lowers UDS for CenterNet | UDS(+DANN) < UDS(sim) on both targets |  | yes |
| text|GUPNet DANN MAV | 22.05 | 22.05 | yes |
| text|GUPNet DANN RD | 14.34 | 14.34 | yes |
| text|PST-ST MAV min | 9.6 | 9.6 | yes |
| text|PST-ST MAV max | 39.1 | 39.1 | yes |
| text|PST-ST RD min | 3.3 | 3.3 | yes |
| text|PST-ST RD max | 23.1 | 23.1 | yes |
| text|CenterNet* ASE | 1.74 | 1.74 | yes |
| text|CenterNet ASE | 5.92 | 5.92 | yes |
| text|CenterNet* ATE | 0.283 | 0.283 | yes |
| text|CenterNet ATE | 0.666 | 0.666 | yes |
| text|full pipeline gain mav | 11.3 | 11.3 | yes |
| text|conclusion gain mav | 11.3 | 11.3 | yes |
| text|full pipeline gain anti | 8.1 | 8.1 | yes |
| text|conclusion gain anti | 8.1 | 8.1 | yes |
| text|full pipeline UDS MAV | 60.81 | 60.81 | yes |
| text|full pipeline UDS RD | 38.69 | 38.69 | yes |
| text|AP CenterNet* MAV | 26.87 | 26.87 | yes |
| text|AP full MAV | 40.93 | 40.93 | yes |
| text|AP CenterNet* RD | 16.04 | 16.04 | yes |
| text|AP full RD | 23.28 | 23.28 | yes |
| text|full pipeline detects more and lower AOE (both targets) | mDR up and AOE down vs CenterNet* |  | yes |
| text|full pipeline best UDS of Tab. 1 on RD | max UDS on RD |  | yes |
| text|within 4.6 of real labels | 4.6 | 4.5 | **NO** |
| text|MonoCoP +PST MAV UDS | 61.90 | 61.90 | yes |
| text|MonoDGP +PST MAV UDS | 60.55 | 60.55 | yes |
| text|PST - w/o sensors mav | 4.22 | 4.22 | yes |
| text|conclusion PST - w/o sensors mav | 4.2 | 4.2 | yes |
| text|PST - w/o sensors anti | 2.47 | 2.47 | yes |
| text|conclusion PST - w/o sensors anti | 2.5 | 2.5 | yes |
| text|95% interval excludes zero mav (read from saved bootstrap json) | interval of PST - w/o sensors excludes zero | saved: [-6.671124491006582, -2.2611529783667925] | yes |
| text|95% interval excludes zero anti (read from saved bootstrap json) | interval of PST - w/o sensors excludes zero | saved: [-3.7216377891465338, -1.1285085457151804] | yes |
| text|PST above all controls on both targets | mean PST > every control mean |  | yes |
| text|furthest behind on RD (up to) | 6.87 | 6.87 | yes |
| text|PST RD all frames | 41.71 | 41.71 | yes |
| text|PST RD restricted | 38.69 | 38.69 | yes |
| text|tab2 mDR row1 | 69.77 | 69.77 | yes |
| text|tab2 mDR step1 | 96.23 | 96.23 | yes |
| text|tab2 AOE step2 | 14.48 | 14.48 | yes |
| text|tab2 AOE row1 | 20.32 | 20.32 | yes |
| text|tab2 AP step2 | 34.04 | 34.04 | yes |
| text|tab2 AP step3 | 40.93 | 40.93 | yes |
| text|full ATE lower on as many frames as CenterNet* detects | ATE(row 6) < ATE(row 1) | 0.247 vs 0.283 | yes |
| text|composites: UDS from | 60.81 | 60.81 | yes |
| text|composites: UDS to | 55.32 | 55.32 | yes |
| text|abstract CenterNet PST-sim mav | 14.5 | 14.5 | yes |
| text|abstract CenterNet PST-sim anti | 12.8 | 12.8 | yes |
| text|fig4 det depth err | 7.6 | 7.6 | yes |
| text|fig4 PST depth err | 5.9 | 5.9 | yes |
| text|fig4 RGB-only depth err | 8.8 | 8.8 | yes |
| text|fig4 det orient err | 30.1 | 30.1 | yes |
| text|fig4 PST orient err | 18.8 | 18.8 | yes |

# Datasets

The study uses nine public inertial HAR datasets. **No dataset files are redistributed in this repository.** The
notebook downloads each dataset from its original source (URLs in `DATASET_REGISTRY`, notebook Section 2). Each
dataset stays under the terms of its original repository: check the linked page before use, and cite the
original paper.

| Key | Dataset | Split | Subjects | Labels | Channels | Rate (Hz) | Acc. unit | Gyro. unit | Source | Original paper (DOI) |
|---|---|---|--:|--:|--:|--:|---|---|---|---|
| `hapt` | HAPT (smartphone, postural transitions) | train | 30 | 12 | 6 | 50 | g | rad/s | [UCI 341](https://archive.ics.uci.edu/dataset/341/smartphone+based+recognition+of+human+activities+and+postural+transitions) | [10.1016/j.neucom.2015.07.085](https://doi.org/10.1016/j.neucom.2015.07.085) |
| `pamap2` | PAMAP2 | train | 9 | 12 | 18 | 100 | m/s² | rad/s | [UCI 231](https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring) | [10.1109/ISWC.2012.13](https://doi.org/10.1109/ISWC.2012.13) |
| `wisdm19` | WISDM 2019 (phone accelerometer) | train | 51 | 18 | 3 | 20 | m/s² | – | [UCI 507](https://archive.ics.uci.edu/dataset/507/wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset) | [10.1109/ACCESS.2019.2940729](https://doi.org/10.1109/ACCESS.2019.2940729) |
| `dsads` | DSADS | train | 8 | 19 | 30 | 25 | m/s² | rad/s | [UCI 256](https://archive.ics.uci.edu/dataset/256/daily+and+sports+activities) | [10.1016/j.patcog.2010.04.019](https://doi.org/10.1016/j.patcog.2010.04.019) |
| `realdisp` | REALDISP (ideal placement) | train | 17 | 33 | 54 | 50 | m/s² | rad/s | [UCI 305](https://archive.ics.uci.edu/dataset/305/realdisp+activity+recognition+dataset) | [10.1145/2370216.2370437](https://doi.org/10.1145/2370216.2370437) |
| `mhealth` | MHEALTH | held out | 10 | 12 | 15 | 50 | m/s² | deg/s* | [UCI 319](https://archive.ics.uci.edu/dataset/319/mhealth+dataset) | [10.1007/978-3-319-13105-4_14](https://doi.org/10.1007/978-3-319-13105-4_14) |
| `uschad` | USC-HAD | held out | 14 | 12 | 6 | 100 | g | deg/s | [USC SIPI](http://sipi.usc.edu/had/) | [10.1145/2370216.2370438](https://doi.org/10.1145/2370216.2370438) |
| `harth` | HARTH | held out | 22 | 12 | 6 | 50 | g | – | [UCI 779](https://archive.ics.uci.edu/dataset/779/harth) | [10.3390/s21237853](https://doi.org/10.3390/s21237853) |
| `motionsense` | MotionSense | held out | 24 | 6 | 6 | 50 | g | rad/s | [GitHub](https://github.com/mmalekzadeh/motion-sense) | [10.1145/3302505.3310068](https://doi.org/10.1145/3302505.3310068) |

Units and rates are taken from the gold records. Channels are the inertial channels the loader uses.

\* MHEALTH documents its gyroscope in deg/s. The gyroscope probe finds a 99th-percentile angular rate of about
1.0, which is typical of rad/s. We report this as a candidate documentation–data discrepancy (medium
confidence); the gold record keeps the documented value.

HHAR ([UCI 344](https://archive.ics.uci.edu/dataset/344/heterogeneity+activity+recognition)) is listed in the notebook but failed to load and is not part of the reported results.

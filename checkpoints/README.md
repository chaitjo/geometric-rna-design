# gRNAde checkpoints

Checkpoints are available for various training data splits.
For general usage and best performance, we recommend using the `all` split.

## All RNAsolo samples (as of October 2023): `all`

(Sequence recovery and Self-consistency MCC are computed on Single-state benchmark from Das et al.)

| Model | Max. no. conformers | Max. training RNA length | Link | Sequence recovery | Self-consistency MCC |
| --- | --- | --- | --- | --- | --- |
| Autoregressive | 1 | 5000 | [gRNAde_ARv1_1state_all.h5](gRNAde_ARv1_1state_all.h5) | 0.7387 | 0.6296 |
| Autoregressive | 2 | 5000 | [gRNAde_ARv1_2state_all.h5](gRNAde_ARv1_2state_all.h5) | 0.7907 | 0.6192 |
| Autoregressive | 3 | 5000 | [gRNAde_ARv1_3state_all.h5](gRNAde_ARv1_3state_all.h5) | 0.7987 | 0.5911 |
| Autoregressive | 5 | 5000 | [gRNAde_ARv1_5state_all.h5](gRNAde_ARv1_5state_all.h5) | 0.8197 | 0.6344 |
| | | | |

## Single-state split from [Das et al., 2010](https://www.nature.com/articles/nmeth.1433): `das`

| Model | Max. no. conformers | Max. training RNA length | Link | Sequence recovery | Self-consistency MCC |
| --- | --- | --- | --- | --- | --- |
| Autoregressive | 1 | 5000 | [gRNAde_ARv1_1state_das.h5](gRNAde_ARv1_1state_das.h5) | 0.5278 | 0.6304 |
| Autoregressive | 2 | 5000 | [gRNAde_ARv1_2state_das.h5](gRNAde_ARv1_2state_das.h5) | 0.5140 | 0.6163 |
| Autoregressive | 3 | 5000 | [gRNAde_ARv1_3state_das.h5](gRNAde_ARv1_3state_das.h5) | 0.5424 | 0.6204 |
| Autoregressive | 5 | 5000 | [gRNAde_ARv1_5state_das.h5](gRNAde_ARv1_5state_das.h5) | 0.5669 | 0.6296 |
| | | | |

## Multi-state split of structurally flexible RNAs: `multi`

| Model | Max. no. conformers | Max. training RNA length | Link | Sequence recovery | Self-consistency MCC |
| --- | --- | --- | --- | --- | --- |
| Autoregressive | 1 | 5000 | [gRNAde_ARv1_1state_multi.h5](gRNAde_ARv1_1state_multi.h5) | 0.4468 | 0.5809 |
| Autoregressive | 2 | 5000 | [gRNAde_ARv1_2state_multi.h5](gRNAde_ARv1_2state_multi.h5) | 0.4471 | 0.5321 |
| Autoregressive | 3 | 5000 | [gRNAde_ARv1_3state_multi.h5](gRNAde_ARv1_3state_multi.h5) | 0.4692 | 0.5957 |
| Autoregressive | 5 | 5000 | [gRNAde_ARv1_5state_multi.h5](gRNAde_ARv1_5state_multi.h5) | 0.4844 | 0.5841 |
| | | | |

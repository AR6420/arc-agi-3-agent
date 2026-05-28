# Phase 2.5 Inference-Policy Sweep Log


## Run 2026-05-28T00:14:49

_Experiment 1: action-type temperature sweep_


| config | kwargs | train | holdout | nz train | nz holdout | wall |
|---|---|---:|---:|---:|---:|---:|
| argmax_baseline | `—` | 0.924 | **0.000** | 2/20 | 0/5 | 6.0s |
| T0.5 | `temperature=0.5` | 1.436 | **0.000** | 5/20 | 0/5 | 7.5s |
| T1.0 | `temperature=1.0` | 0.805 | **0.000** | 2/20 | 0/5 | 6.5s |
| T1.5 | `temperature=1.5` | 0.119 | **0.000** | 2/20 | 0/5 | 6.3s |
| T2.0 | `temperature=2.0` | 0.091 | **0.000** | 1/20 | 0/5 | 6.0s |

## Run 2026-05-28T00:15:55

_Experiment 2: spatial top-k on top of T0.5_


| config | kwargs | train | holdout | nz train | nz holdout | wall |
|---|---|---:|---:|---:|---:|---:|
| T0.5_topk5 | `temperature=0.5, spatial_topk=5` | 2.236 | **0.000** | 9/20 | 0/5 | 17.2s |
| T0.5_topk20 | `temperature=0.5, spatial_topk=20` | 1.333 | **0.000** | 8/20 | 0/5 | 13.1s |
| T0.5_topk50 | `temperature=0.5, spatial_topk=50` | 1.065 | **0.000** | 8/20 | 0/5 | 8.7s |

## Run 2026-05-28T00:17:16

_Experiment 3+4: framechange filter + stuck detector_


| config | kwargs | train | holdout | nz train | nz holdout | wall |
|---|---|---:|---:|---:|---:|---:|
| T0.5_topk5_fc | `temperature=0.5, spatial_topk=5, framechange_filter=True` | 2.124 | **0.000** | 4/20 | 0/5 | 19.0s |
| T0.5_topk5_stuck8 | `temperature=0.5, spatial_topk=5, stuck_detector_K=8` | 3.552 | **0.000** | 5/20 | 0/5 | 9.3s |
| T0.5_topk5_fc_stuck8 | `temperature=0.5, spatial_topk=5, framechange_filter=True, stuck_detector_K=8` | 2.134 | **0.000** | 5/20 | 0/5 | 20.2s |

## Run 2026-05-28T02:42:17

_Phase 2.5 Part 4 — v2 weights (3.6M params, spatial_w=0.6, 38ep)_


| config | kwargs | train | holdout | nz train | nz holdout | wall |
|---|---|---:|---:|---:|---:|---:|
| argmax_baseline | `—` | 0.139 | **0.000** | 1/20 | 0/5 | 10.2s |
| T0.5 | `temperature=0.5` | 1.600 | **0.000** | 5/20 | 0/5 | 15.3s |
| T1.0 | `temperature=1.0` | 0.675 | **0.000** | 2/20 | 0/5 | 13.3s |
| T1.5 | `temperature=1.5` | 0.000 | **0.000** | 0/20 | 0/5 | 12.2s |
| T2.0 | `temperature=2.0` | 0.000 | **0.000** | 0/20 | 0/5 | 11.9s |
| T0.5_topk5 | `temperature=0.5, spatial_topk=5` | 1.666 | **0.000** | 6/20 | 0/5 | 15.4s |
| T0.5_topk20 | `temperature=0.5, spatial_topk=20` | 1.866 | **0.000** | 6/20 | 0/5 | 15.8s |
| T0.5_topk50 | `temperature=0.5, spatial_topk=50` | 1.627 | **0.000** | 5/20 | 0/5 | 15.4s |
| T0.5_topk5_fc | `temperature=0.5, spatial_topk=5, framechange_filter=True` | 1.248 | **0.000** | 5/20 | 0/5 | 26.4s |
| T0.5_topk5_stuck8 | `temperature=0.5, spatial_topk=5, stuck_detector_K=8` | 1.619 | **0.000** | 6/20 | 0/5 | 14.7s |
| T0.5_topk5_fc_stuck8 | `temperature=0.5, spatial_topk=5, framechange_filter=True, stuck_detector_K=8` | 1.248 | **0.000** | 5/20 | 0/5 | 26.6s |

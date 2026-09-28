# Generalisation experiments

DEV = validation + future malware reported before 2026-01-01; FINAL = test benign + future malware from 2026-01-01 + live PyPI. Thresholds come from grouped CV on the training split only.

```
config                   C0 v1 features (shipped before)  C1 v4 features  C2 v4 + monotone (chosen)  C3 v4 + monotone - shape
n_features                                       69.0000        108.0000                   108.0000                   99.0000
oof_pr_auc                                        0.9978          0.9984                     0.9983                    0.9982
dev_acc@f05                                       0.9273          0.9535                     0.9538                    0.9415
dev_prec@f05                                      0.9965          0.9991                     0.9986                    0.9976
dev_rec@f05                                       0.8942          0.9322                     0.9331                    0.9158
dev_fpr@f05                                       0.0064          0.0018                     0.0027                    0.0045
final_acc@f05                                     0.8436          0.8819                     0.8783                    0.8629
final_prec@f05                                    0.9706          0.9737                     0.9709                    0.9736
final_rec@f05                                     0.5288          0.6502                     0.6406                    0.5895
final_fpr@f05                                     0.0076          0.0083                     0.0091                    0.0076
dev_future_rec@f05                                0.7245          0.8010                     0.8073                    0.7596
final_future_rec@f05                              0.5477          0.6728                     0.6628                    0.6093
final_live_mal_rec@f05                            0.1111          0.1481                     0.1481                    0.1481
final_live_ben_fpr@f05                            0.0091          0.0136                     0.0182                    0.0273
dev_acc@fpr1                                      0.9467          0.9579                     0.9596                    0.9462
dev_prec@fpr1                                     0.9941          0.9973                     0.9973                    0.9949
dev_rec@fpr1                                      0.9255          0.9404                     0.9430                    0.9253
dev_fpr@fpr1                                      0.0109          0.0054                     0.0054                    0.0100
final_acc@fpr1                                    0.8704          0.8886                     0.8891                    0.8722
final_prec@fpr1                                   0.9537          0.9637                     0.9702                    0.9654
final_rec@fpr1                                    0.6266          0.6789                     0.6757                    0.6246
final_fpr@fpr1                                    0.0144          0.0121                     0.0098                    0.0106
dev_future_rec@fpr1                               0.7866          0.8232                     0.8344                    0.7850
final_future_rec@fpr1                             0.6449          0.7028                     0.6945                    0.6461
final_live_mal_rec@fpr1                           0.2222          0.1481                     0.2593                    0.1481
final_live_ben_fpr@fpr1                           0.0136          0.0273                     0.0182                    0.0273
thr_f05                                           0.8200          0.6000                     0.5400                    0.5900
thr_fpr1                                          0.6153          0.3777                     0.4003                    0.4312
```

## Monthly retraining, decision rules (new-family malware vs ~1,100 benign)

Choose (2025-07..12):

```
                          accuracy  precision  recall     fpr
policy                                                       
model @F0.5                 0.9804     0.9275  0.8041  0.0051
model @1% FPR               0.9788     0.8762  0.8373  0.0097
model @0.5% FPR             0.9813     0.9284  0.8152  0.0051
hard gate @1% FPR           0.9741     0.9363  0.7061  0.0039
soft gate @1% / strict      0.9806     0.9370  0.7967  0.0044
soft gate @F0.5 / strict    0.9811     0.9593  0.7837  0.0027
soft gate @0.5% / strict    0.9808     0.9570  0.7819  0.0029
```

Report (2026-01..08):

```
                          accuracy  precision  recall     fpr
policy                                                       
model @F0.5                 0.9768     0.8874  0.7118  0.0059
model @1% FPR               0.9751     0.8314  0.7448  0.0099
model @0.5% FPR             0.9775     0.9101  0.7031  0.0045
hard gate @1% FPR           0.9774     0.9233  0.6892  0.0037
soft gate @1% / strict      0.9795     0.9246  0.7240  0.0039
soft gate @F0.5 / strict    0.9789     0.9458  0.6962  0.0026
soft gate @0.5% / strict    0.9792     0.9590  0.6910  0.0019
```

# Model A ablations

Future cutoff: 2025-06-01, 1225 future packages. Test: 2939 packages.

```
config              A3 new features, new data  A5 = A3 + synthetic
oof_pr_auc                             0.9988               0.9985
test_pr_auc                            0.9960               0.9937
thr_f05                                0.7500               0.7800
precision_f05                          0.9963               0.9970
recall_f05                             0.9230               0.9203
fpr_f05                                0.0034               0.0027
obscure_fpr_f05                        0.0093               0.0056
recall_fpr1                            0.9567               0.9292
fpr_fpr1                               0.0115               0.0061
recall_big_f05                         0.6164               0.7534
recall_big_fpr1                        0.6849               0.7534
recall_datadog_f05                     0.8015               0.7893
recall_malreg_f05                      0.9712               0.9722
future_recall_f05                      0.6580               0.6759
future_recall_fpr1                     0.7543               0.7061
probe_index-forum                      0.0018               0.0258
```

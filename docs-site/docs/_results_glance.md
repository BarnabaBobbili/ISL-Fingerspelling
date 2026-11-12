| Task | Test | Result |
|---|---|---|
| Static letters (DH-GAT) | real-video frames, held-out videos | **95.9 %** (baseline: 38.4 %) |
| Static letters | original in-domain split | **98.0 %** |
| Static letters | cross-dataset (Static Dataset B never seen) | **70.0 %** |
| Calibration | expected calibration error on real-video frames | **0.011** (baseline: 0.170) |
| Continuous (Conformer-CTC) | character error rate, test videos | **3.63 %** (baseline BiLSTM: 14.75 %) |
| Contact alignment of the 3D hands | mean contact gap | **0.214 → 0.017** hand sizes |

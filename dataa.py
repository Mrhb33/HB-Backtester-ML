[QT-VALIDATE] qtStart=50453 qtEnd=55709 warmup=220 seriesLen=72832
[QT-VALIDATE] qtBars=5256 qtWarmStart=50233 barsInSlice=5476
[QT] window: bars[50453:55709] (5256 bars) warmup_start=50233 ts=1685145600000..1704063600000 warmup=220
[QuickTest] result: trds=16 ret=-0.0566 pf=0.85 dd=0.116 fee=20 slip=5
[STABILITY-PENALTY] val_score=-0.3673->-0.1837 stab_ratio=-0.01 penalty=0.50
[STABILITY-PENALTY] val_score=-1.3887->-0.6944 stab_ratio=-0.04 penalty=0.50
[STABILITY-PENALTY] val_score=-0.4679->-0.2339 stab_ratio=-0.01 penalty=0.50
16:41:34Z  [PROG]  Batch 9: Tested 82048 | Train: 33.5488 | batchValScore: -0.3673 | Ret: 1.62% | WR: 40.7% | Trds: 150 | Rate: 788/s | Runtime: 1m37s | fp: (LT F[35] -0.59)|(LT F[20] -85.05)|(AND (SlopeGT F[31] 0.00 8) (GT F[12] 57.12))|ATR*4.00|ATR*14.22|ATR*4.82|hold=528|loss=20|cd=150|fees=20.0000|slip=5.0000|risk=1.0000|dir=1|vol_ATR14>SMA36*0.97| ✗
  [crit: score>-5.00, trds>=30, DD<0.36(36.0%), ret>0.5%, exp>0.0001, pf>1.03, stab>50%]
  [val: score=-0.3673 dd=29.67% trds=150 reason=stab=-1%]
  [QuickTest: Score=-5.8198 Ret=-5.66% DD=11.57% Trds=16]
  [train: score=33.55 ret=372.7% dd=16.5% exp=0.0189 pf=2.35 trds=92 geo=2.92%/mo months=54]
│ bestValScore=6.5747 =                            ││ stagnation=161                                   │
    Regime Filter (must be true):     (AND (SlopeGT MinusDI(F[31]) 0.00 8) (GT RSI14(F[12]) 57.12))
    Entry Signal (candle t close):     (LT VolZ20(F[35]) -0.59)
    Entry Execution (candle t+1 open): pending entry enters at next bar open
    Exit Signal:     (LT MACD(F[20]) -85.05)
  [RISK MANAGEMENT]
    StopLoss: ATR*4.00
    TakeProfit: ATR*14.22
    Trail: ATR*4.82
  [WALK-FORWARD OOS RESULTS]
    GeoAvg: 1.62% | Median: 0.94% | MinMonth: -10.83% | Months: 74 | Trades: 150 | MaxDD: 29.67%
    Sparse Months (0 trades): 1.4% (1/74 months)
    Monthly Breakdown (first 6 months shown):
      M00: Ret=-5.26% DD=5.45% Trades=1
      M01: Ret=26.48% DD=6.62% Trades=1
      M02: Ret=4.46% DD=7.84% Trades=1
      M03: Ret=13.89% DD=10.61% Trades=3
      M04: Ret=10.50% DD=6.64% Trades=1
      M05: Ret=12.09% DD=3.32% Trades=2
      ... and 68 more months (see best_every_10000.txt for full breakdown)

Tested: 83,968 | Rate: 768/s | Best: 0.1500 | Elites: 30 | Gen: 619
gen=116979 rej_sur=0(0.0%) rej_seen=13213/130234(10.1%) rerolled=13155(11.2%) rej_novelty=0(0.0%) sent=92785(79.3%)[SAMPLED REJECTION #90001] reason=wf_fold_entry_rate_too_low: edges=0, zero_folds=74/74, edge_rate=0.0/year < 0.5, seed=4141982429625323258, trades=0, ret=0.00%, pf=0.00, exp=0.00000, dd=0.00%
Tested: 91,008 | Rate: 704/s | Best: 0.1500 | Elites: 30 | Gen: 619
gen=126188 rej_sur=0(0.0%) rej_seen=15419/141632(10.9%) rerolled=15344(12.2%) rej_novelty=0(0.0%) sent=100033(79.3%)[QT-VALIDATE] qtStart=50453 qtEnd=55709 warmup=214 seriesLen=72832
[QT-VALIDATE] qtBars=5256 qtWarmStart=50239 barsInSlice=5470
[QT] window: bars[50453:55709] (5256 bars) warmup_start=50239 ts=1685145600000..1704063600000 warmup=214
[QuickTest] result: trds=18 ret=0.0637 pf=1.27 dd=0.113 fee=20 slip=5
[STABILITY-PENALTY] val_score=-0.9406->-0.4703 stab_ratio=-0.02 penalty=0.50
[STABILITY-PENALTY] val_score=-1.1508->-0.5754 stab_ratio=-0.04 penalty=0.50
[STABILITY-PENALTY] val_score=-1.2848->-0.6424 stab_ratio=-0.02 penalty=0.50
[STABILITY-PENALTY] val_score=-1.3314->-0.6657 stab_ratio=-0.04 penalty=0.50
[STABILITY-PENALTY] val_score=-1.4759->-0.7379 stab_ratio=-0.10 penalty=0.50
[STABILITY-PENALTY] val_score=-1.3829->-0.6915 stab_ratio=-0.03 penalty=0.50
[STABILITY-PENALTY] val_score=-1.5694->-0.7847 stab_ratio=-0.03 penalty=0.50
[STABILITY-PENALTY] val_score=-1.3300->-0.6650 stab_ratio=-0.03 penalty=0.50
[STABILITY-PENALTY] val_score=-1.5633->-0.7816 stab_ratio=-0.03 penalty=0.50
[STABILITY-PENALTY] val_score=-1.3985->-0.6993 stab_ratio=-0.03 penalty=0.50
[STABILITY-PENALTY] val_score=-1.6182->-0.8091 stab_ratio=-0.03 penalty=0.50
[STABILITY-PENALTY] val_score=-1.6182->-0.8091 stab_ratio=-0.03 penalty=0.50
[STABILITY-PENALTY] val_score=-1.3194->-0.6597 stab_ratio=-0.03 penalty=0.50
[STABILITY-PENALTY] val_score=-1.5464->-0.7732 stab_ratio=-0.03 penalty=0.50
[STABILITY-PENALTY] val_score=-1.7378->-0.8689 stab_ratio=-0.04 penalty=0.50
[STABILITY-PENALTY] val_score=-1.7419->-0.8710 stab_ratio=-0.04 penalty=0.50
[STABILITY-PENALTY] val_score=-1.8144->-0.9072 stab_ratio=-0.04 penalty=0.50
[STABILITY-PENALTY] val_score=-1.7841->-0.8921 stab_ratio=-0.05 penalty=0.50
[STABILITY-PENALTY] val_score=-1.9137->-0.9569 stab_ratio=-0.05 penalty=0.50
[STABILITY-PENALTY] val_score=-1.8674->-0.9337 stab_ratio=-0.05 penalty=0.50
16:41:48Z  [PROG]  Batch 10: Tested 92032 | Train: 30.2161 | batchValScore: -0.9406 | Ret: 1.48% | WR: 38.6% | Trds: 189 | Rate: 727/s | Runtime: 1m51s | fp: (GT F[12] 57.60)||(AND (SlopeGT F[31] 0.00 8) (GT F[12] 57.12))|ATR*6.93|ATR*11.73|ATR*4.82|hold=528|loss=20|cd=150|fees=20.0000|slip=5.0000|risk=1.0000|dir=1|vol_ATR14>SMA36*0.97| ✗
  [crit: score>-5.00, trds>=30, DD<0.37(37.0%), ret>0.0%, exp>0.0001, pf>1.01, stab>50%]
  [val: score=-0.9406 dd=29.42% trds=189 reason=stab=-3%]
  [QuickTest: Score=0.0153 Ret=6.37% DD=11.29% Trds=18]
│ Stats: passRate=0.01% │ elites=30 │              ││ Meta: radicalP=0.61 │ surExploreP=0.65 │         │
│ bestValScore=6.5747 =                            ││ stagnation=162                                   │
    Regime Filter (must be true):     (AND (SlopeGT MinusDI(F[31]) 0.00 8) (GT RSI14(F[12]) 57.12))
    Entry Signal (candle t close):     (GT RSI14(F[12]) 57.60)
    Entry Execution (candle t+1 open): pending entry enters at next bar open
│ Stats: passRate=0.01% │ elites=30 │              ││ Meta: radicalP=0.63 │ surExploreP=0.70 │         │
  [RISK MANAGEMENT]
    StopLoss: ATR*6.93
    TakeProfit: ATR*11.73
    Trail: ATR*4.82
  [WALK-FORWARD OOS RESULTS]
    GeoAvg: 1.48% | Median: -0.56% | MinMonth: -10.89% | Months: 74 | Trades: 189 | MaxDD: 29.42%
    Sparse Months (0 trades): 0.0% (0/74 months)
    Monthly Breakdown (first 6 months shown):
      M00: Ret=14.49% DD=10.54% Trades=2
      M01: Ret=13.39% DD=9.43% Trades=2
      M02: Ret=3.05% DD=9.09% Trades=1
      M03: Ret=10.90% DD=13.25% Trades=3
      M04: Ret=10.45% DD=7.21% Trades=1
      M05: Ret=14.54% DD=4.07% Trades=2
      ... and 68 more months (see best_every_10000.txt for full breakdown)

Tested: 98,432 | Rate: 742/s | Best: 0.1500 | Elites: 30 | Gen: 620
gen=135826 rej_sur=0(0.0%) rej_seen=17818/153651(11.6%) rerolled=17725(13.0%) rej_novelty=0(0.0%) sent=107548(79.2%)Gen: 135,826 (heavy_mut))
Rejects: entry_low=78,878, dd_high=18,848 | Pass: 0.0%

✓ Checkpoint saved: checkpoint.json (runtime: 2m)
[SAMPLED REJECTION #100001] reason=wf_fold_entry_rate_too_low: edges=2, zero_folds=72/74, edge_rate=0.3/year < 0.5, seed=2303550870282690567, trades=2, ret=0.00%, pf=0.00, exp=0.00000, dd=0.00%
[QT-VALIDATE] qtStart=50453 qtEnd=55709 warmup=214 seriesLen=72832
[QT-VALIDATE] qtBars=5256 qtWarmStart=50239 barsInSlice=5470
[QT] window: bars[50453:55709] (5256 bars) warmup_start=50239 ts=1685145600000..1704063600000 warmup=214
[QuickTest] result: trds=18 ret=0.0112 pf=1.08 dd=0.147 fee=20 slip=5
[STABILITY-PENALTY] val_score=-0.5675->-0.2838 stab_ratio=-0.01 penalty=0.50
[QT-VALIDATE] qtStart=50453 qtEnd=55709 warmup=214 seriesLen=72832
[QT-VALIDATE] qtBars=5256 qtWarmStart=50239 barsInSlice=5470
[QT] window: bars[50453:55709] (5256 bars) warmup_start=50239 ts=1685145600000..1704063600000 warmup=214
[QuickTest] result: trds=17 ret=0.2795 pf=2.07 dd=0.154 fee=20 slip=5
[STABILITY-PENALTY] val_score=0.0127->0.0063 stab_ratio=0.00 penalty=0.50
[STABILITY-PENALTY] val_score=-0.1563->-0.0781 stab_ratio=-0.00 penalty=0.50
[CPCV-PENALTY] val_score=-0.0781->-0.0599 cpcv_stab=0.17 penalty=0.77
[QT-VALIDATE] qtStart=50453 qtEnd=55709 warmup=210 seriesLen=72832
[QT-VALIDATE] qtBars=5256 qtWarmStart=50243 barsInSlice=5466
[QT] window: bars[50453:55709] (5256 bars) warmup_start=50243 ts=1685145600000..1704063600000 warmup=210
[QuickTest] result: trds=21 ret=-0.0657 pf=0.81 dd=0.195 fee=20 slip=5
[STABILITY-PENALTY] val_score=-1.3960->-0.6980 stab_ratio=-0.03 penalty=0.50
[18:40:42] ✓ New elite added (score=-0.9204, total=23)                                                                                                                                         
[18:40:42] ✓ New elite added (score=-1.0007, total=24)                                                                                                                                         
[18:40:42] ✓ New elite added (score=-1.0035, total=25)                                                                                                                                         
[18:40:42] ✓ New elite added (score=-1.0388, total=26)                                                                                                                                         
[18:40:42] ✓ New elite added (score=-1.0007, total=24)                                                                                                                                         
[18:40:42] ✓ New elite added (score=-1.0035, total=25)                                                                                                                                         
[18:40:42] ✓ New elite added (score=-1.0388, total=26)                                                                                                                                         
[18:40:42] ✓ New elite added (score=-1.0440, total=27)                                                                                                                                         
[18:40:42] ✓ New elite added (score=-1.1640, total=28)                                                                                                                                         
[18:40:42] ✓ New elite added (score=-1.3246, total=29)                                                                                                                                         
[18:40:42] ✓ New elite added (score=-1.4648, total=30)                                                                                                                                         
[18:42:02] ✓ New elite added (score=-0.0599, total=31)                                                                                                                                         
[18:42:02] ↗ Best score improved: -0.2393 → -0.0599                                                                                                                                            
[18:42:02] ✓ New elite added (score=-0.6033, total=32)                                                                                                                                         
[STABILITY-PENALTY] val_score=-1.6787->-0.8393 stab_ratio=-0.03 penalty=0.50
[STABILITY-PENALTY] val_score=-2.0034->-1.0017 stab_ratio=-0.05 penalty=0.50
[STABILITY-PENALTY] val_score=-1.3557->-0.6779 stab_ratio=-0.02 penalty=0.50
[STABILITY-PENALTY] val_score=-1.8554->-0.9277 stab_ratio=-0.04 penalty=0.50
[STABILITY-PENALTY] val_score=-2.0121->-1.0061 stab_ratio=-0.06 penalty=0.50
[STABILITY-PENALTY] val_score=-1.0224->-0.5112 stab_ratio=-0.02 penalty=0.50
[STABILITY-PENALTY] val_score=-1.0160->-0.5080 stab_ratio=-0.02 penalty=0.50
16:42:02Z  [PROG]  Batch 11: Tested 102016 | Train: 33.3194 | batchValScore: 0.0127 | Ret: 2.26% | WR: 39.1% | Trds: 169 | Rate: 722/s | Runtime: 2m5s | fp: (GT F[45] 0.51)||(AND (SlopeGT F[31] 0.00 8) (GT F[12] 57.12))|Fixed 10.00%|Fixed 23.23%|ATR*4.82|hold=528|loss=20|cd=150|fees=20.0000|slip=5.0000|risk=1.0000|dir=1|vol_ATR14>SMA36*0.97| ✗
  [crit: score>-5.00, trds>=30, DD<0.38(38.0%), ret>0.0%, exp>0.0001, pf>1.00, stab>50%]
  [val: score=0.0127 dd=33.48% trds=169 reason=stab=0%]
  [QuickTest: Score=6.8711 Ret=27.95% DD=15.38% Trds=17]
│ Stats: passRate=0.02% │ elites=32 │              ││ Meta: radicalP=0.63 │ surExploreP=0.70 │         │
│ Stats: passRate=0.01% │ elites=32 │              ││ Meta: radicalP=0.63 │ surExploreP=0.70 │         │
    Regime Filter (must be true):     (AND (SlopeGT MinusDI(F[31]) 0.00 8) (GT RSI14(F[12]) 57.12))
    Entry Signal (candle t close):     (GT RangeWidth(F[45]) 0.51)
    Entry Execution (candle t+1 open): pending entry enters at next bar open
    Exit Signal:     (Always Active - No Filter)
  [RISK MANAGEMENT]
    StopLoss: Fixed 10.00%
    TakeProfit: Fixed 23.23%
    Trail: ATR*4.82
  [WALK-FORWARD OOS RESULTS]
    GeoAvg: 2.26% | Median: -0.13% | MinMonth: -14.35% | Months: 74 | Trades: 169 | MaxDD: 33.48%
    Sparse Months (0 trades): 4.1% (3/74 months)
    Monthly Breakdown (first 6 months shown):
      M00: Ret=14.85% DD=10.54% Trades=2
      M01: Ret=16.12% DD=9.43% Trades=2
      M02: Ret=3.05% DD=9.09% Trades=1
      M03: Ret=10.90% DD=13.25% Trades=3
      M04: Ret=10.45% DD=7.21% Trades=1
      M05: Ret=12.76% DD=6.20% Trades=2
      ... and 68 more months (see best_every_10000.txt for full breakdown)

Tested: 105,600 | Rate: 717/s | Best: 0.1500 | Elites: 32 | Gen: 621
[SAMPLED REJECTION #110001] reason=wf_per_fold_entry_rate_too_low: fold=2/74, fold_edges=0, fold_bars=684, fold_edge_rate=0.0/year < 0.5, seed=8868967859640084248, trades=54, ret=0.00%, pf=0.00, exp=0.00000, dd=0.00%
[QT-VALIDATE] qtStart=50453 qtEnd=55709 warmup=214 seriesLen=72832
[QT-VALIDATE] qtBars=5256 qtWarmStart=50239 barsInSlice=5470
[QT] window: bars[50453:55709] (5256 bars) warmup_start=50239 ts=1685145600000..1704063600000 warmup=214
[QuickTest] result: trds=18 ret=0.0434 pf=1.19 dd=0.149 fee=20 slip=5
[STABILITY-PENALTY] val_score=-0.2171->-0.1085 stab_ratio=-0.00 penalty=0.50
[STABILITY-PENALTY] val_score=-1.2653->-0.6326 stab_ratio=-0.02 penalty=0.50
[STABILITY-PENALTY] val_score=-1.4645->-0.7322 stab_ratio=-0.03 penalty=0.50
[STABILITY-PENALTY] val_score=-1.5195->-0.7598 stab_ratio=-0.03 penalty=0.50
[STABILITY-PENALTY] val_score=-1.4879->-0.7439 stab_ratio=-0.03 penalty=0.50
[STABILITY-PENALTY] val_score=-1.4879->-0.7439 stab_ratio=-0.03 penalty=0.50
[STABILITY-PENALTY] val_score=-1.5637->-0.7818 stab_ratio=-0.03 penalty=0.50
[STABILITY-PENALTY] val_score=-1.7147->-0.8573 stab_ratio=-0.04 penalty=0.50
[STABILITY-PENALTY] val_score=-1.8204->-0.9102 stab_ratio=-0.04 penalty=0.50
[STABILITY-PENALTY] val_score=-1.8932->-0.9466 stab_ratio=-0.08 penalty=0.50
[STABILITY-PENALTY] val_score=-1.9125->-0.9563 stab_ratio=-0.04 penalty=0.50
[STABILITY-PENALTY] val_score=-2.2603->-1.1301 stab_ratio=-0.08 penalty=0.50
[STABILITY-PENALTY] val_score=-1.8161->-0.9080 stab_ratio=-0.04 penalty=0.50
[STABILITY-PENALTY] val_score=-2.0672->-1.0336 stab_ratio=-0.06 penalty=0.50
[STABILITY-PENALTY] val_score=-0.4177->-0.2089 stab_ratio=-0.01 penalty=0.50
[STABILITY-PENALTY] val_score=-1.5230->-0.7615 stab_ratio=-0.04 penalty=0.50
[STABILITY-PENALTY] val_score=-1.8192->-0.9096 stab_ratio=-0.05 penalty=0.50
[QT-VALIDATE] qtStart=50453 qtEnd=55709 warmup=221 seriesLen=72832
[QT-VALIDATE] qtBars=5256 qtWarmStart=50232 barsInSlice=5477
[QT] window: bars[50453:55709] (5256 bars) warmup_start=50232 ts=1685145600000..1704063600000 warmup=221
[QuickTest] result: trds=17 ret=0.2628 pf=2.17 dd=0.081 fee=20 slip=5
[STABILITY-PENALTY] val_score=-0.0906->-0.0453 stab_ratio=-0.00 penalty=0.50
[STABILITY-PENALTY] val_score=-0.1989->-0.0995 stab_ratio=-0.00 penalty=0.50
[CPCV-PENALTY] val_score=-0.0995->-0.0763 cpcv_stab=0.17 penalty=0.77
[QT-VALIDATE] qtStart=50453 qtEnd=55709 warmup=220 seriesLen=72832
[QT-VALIDATE] qtBars=5256 qtWarmStart=50233 barsInSlice=5476
[QT] window: bars[50453:55709] (5256 bars) warmup_start=50233 ts=1685145600000..1704063600000 warmup=220
[QuickTest] result: trds=21 ret=-0.0657 pf=0.81 dd=0.195 fee=20 slip=5
[STABILITY-PENALTY] val_score=-0.1362->-0.0681 stab_ratio=-0.00 penalty=0.50
16:42:14Z  [PROG]  Batch 12: Tested 112000 | Train: 30.4334 | batchValScore: -0.0906 | Ret: 2.21% | WR: 36.0% | Trds: 175 | Rate: 807/s | Runtime: 2m17s | fp: (OR (NOT (Rising F[13] 7)) (LT F[13] 56.37))||(AND (SlopeGT F[31] 0.00 8) (GT F[12] 57.12))|ATR*3.70|Fixed 18.50%|ATR*4.82|hold=528|loss=20|cd=150|fees=20.0000|slip=5.0000|risk=1.0000|dir=1|vol_ATR14>SMA36*0.97| ✗
  [crit: score>-5.00, trds>=30, DD<0.39(39.0%), ret>0.0%, exp>0.0001, pf>1.00, stab>50%]
  [val: score=-0.0906 dd=31.92% trds=175 reason=stab=-0%]
  [QuickTest: Score=10.0736 Ret=26.28% DD=8.12% Trds=17]
│ Stats: passRate=0.01% │ elites=33 │              ││ Meta: radicalP=0.63 │ surExploreP=0.70 │         │
│ bestValScore=6.5747 =                            ││ stagnation=164                                   │
    Regime Filter (must be true):     (AND (SlopeGT MinusDI(F[31]) 0.00 8) (GT RSI14(F[12]) 57.12))
    Entry Signal (candle t close):     (OR (NOT (Rising RSI21(F[13]) 7)) (LT RSI21(F[13]) 56.37))
    Entry Execution (candle t+1 open): pending entry enters at next bar open
    Exit Signal:     (Always Active - No Filter)
  [RISK MANAGEMENT]
    StopLoss: ATR*3.70
    TakeProfit: Fixed 18.50%
    Trail: ATR*4.82
  [WALK-FORWARD OOS RESULTS]
    GeoAvg: 2.21% | Median: -0.87% | MinMonth: -11.56% | Months: 74 | Trades: 175 | MaxDD: 31.92%
[18:40:42] ✓ New elite added (score=-1.0035, total=25)
[18:40:42] ✓ New elite added (score=-1.0388, total=26)
[18:40:42] ✓ New elite added (score=-1.0440, total=27)
[18:40:42] ✓ New elite added (score=-1.1640, total=28)
[18:40:42] ✓ New elite added (score=-1.3246, total=29)
[18:40:42] ✓ New elite added (score=-1.4648, total=30)
[18:42:02] ✓ New elite added (score=-0.0599, total=31)
[18:42:02] ↗ Best score improved: -0.2393 → -0.0599
[18:42:02] ✓ New elite added (score=-0.6033, total=32)
[18:42:14] ✓ New elite added (score=-0.0763, total=33)
Tested: 113,920 | Rate: 832/s | Best: 0.1500 | Elites: 33 | Gen: 622
gen=155718 rej_sur=0(0.0%) rej_seen=22551/178231(12.7%) rerolled=22413(14.4%) rej_novelty=0(0.0%) sent=123036(79.0%)[SAMPLED REJECTION #120001] reason=wf_fold_entry_rate_too_low: edges=40, zero_folds=44/74, edge_rate=6.6/year < 0.5, seed=8154292368971267345, trades=40, ret=0.00%, pf=0.00, exp=0.00000, dd=0.00%
Tested: 121,856 | Rate: 794/s | Best: 0.1500 | Elites: 33 | Gen: 622
gen=165916 rej_sur=0(0.0%) rej_seen=25121/190981(13.2%) rerolled=24965(15.0%) rej_novelty=0(0.0%) sent=130972(78.9%)Gen: 165,916 (heavy_mut))
[QT-VALIDATE] qtStart=50453 qtEnd=55709 warmup=221 seriesLen=72832
[QT-VALIDATE] qtBars=5256 qtWarmStart=50232 barsInSlice=5477
[QT] window: bars[50453:55709] (5256 bars) warmup_start=50232 ts=1685145600000..1704063600000 warmup=221
[QuickTest] result: trds=19 ret=0.1344 pf=1.59 dd=0.109 fee=20 slip=5
[STABILITY-PENALTY] val_score=-0.6532->-0.3266 stab_ratio=-0.02 penalty=0.50
[QT-VALIDATE] qtStart=50453 qtEnd=55709 warmup=214 seriesLen=72832
[QT-VALIDATE] qtBars=5256 qtWarmStart=50239 barsInSlice=5470
[QT] window: bars[50453:55709] (5256 bars) warmup_start=50239 ts=1685145600000..1704063600000 warmup=214
[QuickTest] result: trds=18 ret=0.3161 pf=2.32 dd=0.133 fee=20 slip=5
[STABILITY-PENALTY] val_score=-0.6137->-0.3068 stab_ratio=-0.01 penalty=0.50
[STABILITY-PENALTY] val_score=-1.0984->-0.5492 stab_ratio=-0.02 penalty=0.50
[STABILITY-PENALTY] val_score=-1.5463->-0.7731 stab_ratio=-0.03 penalty=0.50
[STABILITY-PENALTY] val_score=-1.8602->-0.9301 stab_ratio=-0.04 penalty=0.50
[STABILITY-PENALTY] val_score=-1.9686->-0.9843 stab_ratio=-0.05 penalty=0.50
[QT-VALIDATE] qtStart=50453 qtEnd=55709 warmup=400 seriesLen=72832
[QT-VALIDATE] qtBars=5256 qtWarmStart=50053 barsInSlice=5656
[QT] window: bars[50453:55709] (5256 bars) warmup_start=50053 ts=1685145600000..1704063600000 warmup=400
[QuickTest] result: trds=22 ret=0.0237 pf=1.14 dd=0.148 fee=20 slip=5
[STABILITY-PENALTY] val_score=-0.5930->-0.2965 stab_ratio=-0.04 penalty=0.50
[STABILITY-PENALTY] val_score=-1.6143->-0.8071 stab_ratio=-0.07 penalty=0.50
[STABILITY-PENALTY] val_score=-1.8073->-0.9036 stab_ratio=-0.04 penalty=0.50
[STABILITY-PENALTY] val_score=-2.2906->-1.1453 stab_ratio=-0.08 penalty=0.50
[STABILITY-PENALTY] val_score=-1.0558->-0.5279 stab_ratio=-0.02 penalty=0.50
[STABILITY-PENALTY] val_score=-1.8089->-0.9044 stab_ratio=-0.05 penalty=0.50
[18:40:42] ✓ New elite added (score=-1.0440, total=27)
[18:40:42] ✓ New elite added (score=-1.1640, total=28)
[18:40:42] ✓ New elite added (score=-1.3246, total=29)
[18:40:42] ✓ New elite added (score=-1.4648, total=30)
[18:42:02] ✓ New elite added (score=-0.0599, total=31)
[18:42:02] ↗ Best score improved: -0.2393 → -0.0599
[18:42:02] ✓ New elite added (score=-0.6033, total=32)
[18:42:14] ✓ New elite added (score=-0.0763, total=33)
[18:40:42] ✓ New elite added (score=-1.1640, total=28)                                                                                                                                         
[18:40:42] ✓ New elite added (score=-1.3246, total=29)                                                                                                                                         
[18:40:42] ✓ New elite added (score=-1.4648, total=30)                                                                                                                                         
[18:42:02] ✓ New elite added (score=-0.0599, total=31)                                                                                                                                         
[18:42:02] ↗ Best score improved: -0.2393 → -0.0599                                                                                                                                            
[18:42:02] ✓ New elite added (score=-0.6033, total=32)                                                                                                                                         
[18:42:14] ✓ New elite added (score=-0.0763, total=33)                                                                                                                                         
[18:42:28] ✓ New elite added (score=-0.0173, total=34)                                                                                                                                         
[18:42:28] ↗ Best score improved: -0.0599 → -0.0173                                                                                                                                            
[18:42:28] ✓ New elite added (score=0.1867, total=35)                                                                                                                                          
 [META: radicalP 0.70->0.60, surExploreP 0.70->0.60]16:42:28Z  [PROG]  Batch 13: Tested 122112 | Train: 21.7226 | batchValScore: 0.4670 | Ret: 2.90% | WR: 44.2% | Trds: 206 | Rate: 755/s | Runtime: 2m30s | fp: (AND (BreakDown F[24] F[25]) (SlopeLT F[13] -0.04 7))||(SlopeGT F[31] 0.00 8)|ATR*6.93|ATR*13.76|ATR*4.82|hold=528|loss=20|cd=150|fees=20.0000|slip=5.0000|risk=1.0000|dir=1|vol_ATR14>SMA36*0.97| ✗
  [crit: score>-5.00, trds>=30, DD<0.40(40.0%), ret>0.0%, exp>0.0001, pf>1.00, stab>50%]
  [val: score=0.4670 dd=29.31% trds=206 reason=stab=2%]
  [QuickTest: Score=-6.6465 Ret=-8.08% DD=20.84% Trds=21]
│ Stats: passRate=0.01% │ elites=35 │              ││ Meta: radicalP=0.63 │ surExploreP=0.70 │         │
│ bestValScore=6.5747 =                            ││ stagnation=165                                   │
    Regime Filter (must be true):     (SlopeGT MinusDI(F[31]) 0.00 8)
    Entry Signal (candle t close):     (AND (BreakDown ROC10(F[24]) ROC20(F[25])) (SlopeLT RSI21(F[13]) -0.04 7))
    Entry Execution (candle t+1 open): pending entry enters at next bar open
    Exit Signal:     (Always Active - No Filter)
  [RISK MANAGEMENT]
    StopLoss: ATR*6.93
    TakeProfit: ATR*13.76
    Trail: ATR*4.82
  [WALK-FORWARD OOS RESULTS]
    GeoAvg: 2.90% | Median: 1.00% | MinMonth: -12.50% | Months: 74 | Trades: 206 | MaxDD: 29.31%
[18:40:42] ✓ New elite added (score=-1.3246, total=29)
[18:40:42] ✓ New elite added (score=-1.4648, total=30)
[18:42:02] ✓ New elite added (score=-0.0599, total=31)
[18:42:02] ↗ Best score improved: -0.2393 → -0.0599
[18:42:02] ✓ New elite added (score=-0.6033, total=32)
[18:42:14] ✓ New elite added (score=-0.0763, total=33)
[18:42:28] ✓ New elite added (score=-0.0173, total=34)
[18:42:28] ↗ Best score improved: -0.0599 → -0.0173
[18:42:28] ✓ New elite added (score=0.1867, total=35)
[18:42:28] ↗ Best score improved: -0.0173 → 0.1867
[MIN-MONTH-REJECT] absolute_floor: MinMonth=-20.18% < Floor=-20.00% (trds=3, config_min=-20.00%)
Tested: 129,536 | Rate: 768/s | Best: 0.4670 | Elites: 35 | Gen: 623
gen=175720 rej_sur=0(0.0%) rej_seen=27581/203227(13.6%) rerolled=27407(15.6%) rej_novelty=0(0.0%) sent=138652(78.9%)[MIN-MONTH-REJECT] absolute_floor: MinMonth=-22.69% < Floor=-20.00% (trds=5[18:40:42] ✓ New elite added (score=-1.4648, total=30)                                                                                                                                         
[18:42:02] ✓ New elite added (score=-0.0599, total=31)                                                                                                                                         
[18:42:02] ↗ Best score improved: -0.2393 → -0.0599                                                                                                                                            
[18:42:02] ✓ New elite added (score=-0.6033, total=32)                                                                                                                                         
[18:42:14] ✓ New elite added (score=-0.0763, total=33)                                                                                                                                         
[18:42:28] ✓ New elite added (score=-0.0173, total=34)                                                                                                                                         
[18:42:28] ↗ Best score improved: -0.0599 → -0.0173                                                                                                                                            
[18:42:28] ✓ New elite added (score=0.1867, total=35)                                                                                                                                          
[18:42:28] ↗ Best score improved: -0.0173 → 0.1867                                                                                                                                             
[18:42:02] ↗ Best score improved: -0.2393 → -0.0599
[18:42:02] ✓ New elite added (score=-0.6033, total=32)
[18:42:14] ✓ New elite added (score=-0.0763, total=33)
[18:42:28] ✓ New elite added (score=-0.0173, total=34)
[18:42:28] ↗ Best score improved: -0.0599 → -0.0173
[18:42:28] ✓ New elite added (score=0.1867, total=35)
[18:42:28] ↗ Best score improved: -0.0173 → 0.1867
[18:42:40] ✓ New elite added (score=0.2467, total=36)
[18:42:40] ↗ Best score improved: 0.1867 → 0.2467
[18:42:40] ✓ New elite added (score=-0.1055, total=37)
[STABILITY-PENALTY] val_score=-1.7289->-0.8644 stab_ratio=-0.03 penalty=0.50
[STABILITY-PENALTY] val_score=-1.5641->-0.7821 stab_ratio=-0.03 penalty=0.50
[STABILITY-PENALTY] val_score=-1.9579->-0.9789 stab_ratio=-0.05 penalty=0.50
[STABILITY-PENALTY] val_score=-1.5985->-0.7993 stab_ratio=-0.03 penalty=0.50
[STABILITY-PENALTY] val_score=-1.6450->-0.8225 stab_ratio=-0.04 penalty=0.50
[STABILITY-PENALTY] val_score=-1.6709->-0.8355 stab_ratio=-0.04 penalty=0.50
[STABILITY-PENALTY] val_score=-1.7515->-0.8757 stab_ratio=-0.04 penalty=0.50
[STABILITY-PENALTY] val_score=-2.0167->-1.0084 stab_ratio=-0.04 penalty=0.50
[STABILITY-PENALTY] val_score=-1.9308->-0.9654 stab_ratio=-0.07 penalty=0.50
[STABILITY-PENALTY] val_score=-2.2079->-1.1040 stab_ratio=-0.06 penalty=0.50
[STABILITY-PENALTY] val_score=-2.1028->-1.0514 stab_ratio=-0.06 penalty=0.50
[STABILITY-PENALTY] val_score=-0.2639->-0.1319 stab_ratio=-0.00 penalty=0.50
[STABILITY-PENALTY] val_score=-1.0739->-0.5370 stab_ratio=-0.02 penalty=0.50
 [META: radicalP 0.60->0.51, surExploreP 0.60->0.54]16:42:40Z  [PROG]  Batch 14: Tested 132096 | Train: 32.9141 | batchValScore: 0.4794 | Ret: 2.52% | WR: 39.3% | Trds: 168 | Rate: 778/s | Runtime: 2m43s | fp: (LT F[49] 0.37)||(AND (CrossDown F[25] F[24]) (Falling F[23] 5))|ATR*6.05|ATR*15.21|ATR*4.82|hold=528|loss=20|cd=150|fees=20.0000|slip=5.0000|risk=1.0000|dir=1|vol_ATR14>SMA36*0.97| ✗
  [crit: score>-5.00, trds>=30, DD<0.40(40.0%), ret>0.0%, exp>0.0001, pf>1.00, stab>50%]
  [val: score=0.4794 dd=27.98% trds=168 reason=stab=1%]
  [QuickTest: Score=-1.6464 Ret=1.92% DD=10.46% Trds=18]
│ Stats: passRate=0.01% │ elites=37 │              ││ Meta: radicalP=0.63 │ surExploreP=0.70 │         │
│ bestValScore=6.5747 =                            ││ stagnation=166                                   │
    Regime Filter (must be true):     (AND (CrossDown ROC20(F[25]) ROC10(F[24])) (Falling OBV(F[23]) 5))
    Entry Signal (candle t close):     (LT ClosePos(F[49]) 0.37)
    Entry Execution (candle t+1 open): pending entry enters at next bar open
    Exit Signal:     (Always Active - No Filter)
  [RISK MANAGEMENT]
    StopLoss: ATR*6.05
    TakeProfit: ATR*15.21
    Trail: ATR*4.82
  [WALK-FORWARD OOS RESULTS]
    GeoAvg: 2.52% | Median: 0.05% | MinMonth: -13.59% | Months: 74 | Trades: 168 | MaxDD: 27.98%
    Sparse Months (0 trades): 1.4% (1/74 months)
    Monthly Breakdown (first 6 months shown):
      M00: Ret=5.97% DD=4.65% Trades=1
      M01: Ret=9.64% DD=9.62% Trades=3
      M02: Ret=10.36% DD=17.72% Trades=2
      M03: Ret=0.00% DD=0.00% Trades=0
      M04: Ret=-7.70% DD=12.79% Trades=2
      M05: Ret=17.81% DD=3.32% Trades=1
      ... and 68 more months (see best_every_10000.txt for full breakdown)

Tested: 136,448 | Rate: 691/s | Best: 0.4794 | Elites: 37 | Gen: 624
gen=184582 rej_sur=0(0.0%) rej_seen=29745/214226(13.9%) rerolled=29544(16.0%) rej_novelty=0(0.0%) sent=145564(78.9%)[SAMPLED REJECTION #140001] reason=wf_oos_rejected: Max DD exceeded: 91.30% > 35.00%, seed=8491185033191619808, trades=445, ret=-3.25%, pf=0.22, exp=-0.00542, dd=91.30%
[QT-VALIDATE] qtStart=50453 qtEnd=55709 warmup=214 seriesLen=72832
[QT-VALIDATE] qtBars=5256 qtWarmStart=50239 barsInSlice=5470
[QT] window: bars[50453:55709] (5256 bars) warmup_start=50239 ts=1685145600000..1704063600000 warmup=214
[QuickTest] result: trds=18 ret=0.0659 pf=1.28 dd=0.114 fee=20 slip=5
[STABILITY-PENALTY] val_score=-0.9610->-0.4805 stab_ratio=-0.02 penalty=0.50
[STABILITY-PENALTY] val_score=-1.2570->-0.6285 stab_ratio=-0.02 penalty=0.50
[STABILITY-PENALTY] val_score=-1.7586->-0.8793 stab_ratio=-0.04 penalty=0.50
[STABILITY-PENALTY] val_score=-1.7775->-0.8887 stab_ratio=-0.03 penalty=0.50
[STABILITY-PENALTY] val_score=-1.9068->-0.9534 stab_ratio=-0.05 penalty=0.50
[STABILITY-PENALTY] val_score=-1.8635->-0.9317 stab_ratio=-0.04 penalty=0.50
[STABILITY-PENALTY] val_score=-1.8723->-0.9361 stab_ratio=-0.04 penalty=0.50
[STABILITY-PENALTY] val_score=-1.8364->-0.9182 stab_ratio=-0.04 penalty=0.50
[STABILITY-PENALTY] val_score=-2.0056->-1.0028 stab_ratio=-0.05 penalty=0.50
[STABILITY-PENALTY] val_score=-2.2181->-1.1091 stab_ratio=-0.08 penalty=0.50
[STABILITY-PENALTY] val_score=-2.1807->-1.0903 stab_ratio=-0.07 penalty=0.50
[QT-VALIDATE] qtStart=50453 qtEnd=55709 warmup=220 seriesLen=72832
[QT-VALIDATE] qtBars=5256 qtWarmStart=50233 barsInSlice=5476
[QT] window: bars[50453:55709] (5256 bars) warmup_start=50233 ts=1685145600000..1704063600000 warmup=220
[QuickTest] result: trds=18 ret=-0.0465 pf=0.91 dd=0.180 fee=20 slip=5
[STABILITY-PENALTY] val_score=-0.1365->-0.0682 stab_ratio=-0.00 penalty=0.50
[STABILITY-PENALTY] val_score=-2.3610->-1.1805 stab_ratio=-0.08 penalty=0.50
[STABILITY-PENALTY] val_score=-0.5102->-0.2551 stab_ratio=-0.01 penalty=0.50
[STABILITY-PENALTY] val_score=-0.6563->-0.3281 stab_ratio=-0.01 penalty=0.50
[STABILITY-PENALTY] val_score=-0.8098->-0.4049 stab_ratio=-0.01 penalty=0.50
[STABILITY-PENALTY] val_score=-1.1033->-0.5517 stab_ratio=-0.02 penalty=0.50
[STABILITY-PENALTY] val_score=-1.3145->-0.6572 stab_ratio=-0.03 penalty=0.50
[STABILITY-PENALTY] val_score=-2.1980->-1.0990 stab_ratio=-0.05 penalty=0.50
[STABILITY-PENALTY] val_score=-1.7311->-0.8655 stab_ratio=-0.04 penalty=0.50
16:42:54Z  [PROG]  Batch 15: Tested 142080 | Train: 35.5082 | batchValScore: -0.1365 | Ret: 2.54% | WR: 43.1% | Trds: 202 | Rate: 756/s | Runtime: 2m56s | fp: (GT F[25] 2.14)||(SlopeGT F[27] -0.06 9)|ATR*6.93|ATR*13.76|ATR*4.82|hold=528|loss=20|cd=150|fees=20.0000|slip=5.0000|risk=1.0000|dir=1|vol_ATR14>SMA36*0.97| ✗
  [crit: score>-5.00, trds>=30, DD<0.40(40.0%), ret>0.0%, exp>0.0001, pf>1.00, stab>50%]
  [val: score=-0.1365 dd=31.28% trds=202 reason=stab=-0%]
  [QuickTest: Score=-5.4180 Ret=-4.65% DD=17.99% Trds=18]
  [train: score=35.51 ret=628.6% dd=20.7% exp=0.0183 pf=2.01 trds=126 geo=3.75%/mo months=54]
│ bestValScore=6.5747 =                            ││ stagnation=167                                   │
    Regime Filter (must be true):     (SlopeGT ATR14(F[27]) -0.06 9)
    Entry Signal (candle t close):     (GT ROC20(F[25]) 2.14)
    Entry Execution (candle t+1 open): pending entry enters at next bar open
│ Stats: passRate=0.01% │ elites=37 │              ││ Meta: radicalP=0.61 │ surExploreP=0.65 │         │
  [RISK MANAGEMENT]
    StopLoss: ATR*6.93
    TakeProfit: ATR*13.76
    Trail: ATR*4.82
  [WALK-FORWARD OOS RESULTS]
    GeoAvg: 2.54% | Median: 0.87% | MinMonth: -13.35% | Months: 74 | Trades: 202 | MaxDD: 31.28%
    Sparse Months (0 trades): 0.0% (0/74 months)
    Monthly Breakdown (first 6 months shown):
      M00: Ret=14.85% DD=10.54% Trades=2
      M01: Ret=36.31% DD=9.43% Trades=3
      M02: Ret=-13.22% DD=25.85% Trades=3
      M03: Ret=19.71% DD=13.22% Trades=3
      M04: Ret=9.93% DD=7.21% Trades=1
      M05: Ret=17.18% DD=7.42% Trades=3
      ... and 68 more months (see best_every_10000.txt for full breakdown)

Tested: 144,256 | Rate: 781/s | Best: 0.4794 | Elites: 37 | Gen: 625
gen=194547 rej_sur=0(0.0%) rej_seen=32308/226738(14.2%) rerolled=32092(16.5%) rej_novelty=0(0.0%) sent=153281(78.8%)Gen: 194,547 (heavy_mut))
Rejects: entry_low=113,816, dd_high=29,389 | Pass: 0.0%

✓ Checkpoint saved: checkpoint.json (runtime: 3m)
[SAMPLED REJECTION #150001] reason=wf_fold_entry_rate_too_low: edges=1, zero_folds=73/74, edge_rate=0.2/year < 0.5, seed=758425996343640670, trades=1, ret=0.00%, pf=0.00, exp=0.00000, dd=0.00%
Tested: 151,680 | Rate: 742/s | Best: 0.4794 | Elites: 37 | Gen: 625
gen=204094 rej_sur=0(0.0%) rej_seen=34783/238747(14.6%) rerolled=34553(16.9%) rej_novelty=0(0.0%) sent=160738(78.8%)[MIN-MONTH-REJECT] absolute_floor: MinMonth=-23.30% < Floor=-20.00% (trds=3, config_min=-20.00%)
[QT-VALIDATE] qtStart=50453 qtEnd=55709 warmup=220 seriesLen=72832
[QT-VALIDATE] qtBars=5256 qtWarmStart=50233 barsInSlice=5476
[QT] window: bars[50453:55709] (5256 bars) warmup_start=50233 ts=1685145600000..1704063600000 warmup=220
[QuickTest] result: trds=21 ret=0.0173 pf=1.12 dd=0.126 fee=20 slip=5
[STABILITY-PENALTY] val_score=-1.0800->-0.5400 stab_ratio=-0.19 penalty=0.50
[QT-VALIDATE] qtStart=50453 qtEnd=55709 warmup=200 seriesLen=72832
[QT-VALIDATE] qtBars=5256 qtWarmStart=50253 barsInSlice=5456
[QT] window: bars[50453:55709] (5256 bars) warmup_start=50253 ts=1685145600000..1704063600000 warmup=200
[QuickTest] result: trds=19 ret=0.0738 pf=1.41 dd=0.102 fee=20 slip=5
[STABILITY-PENALTY] val_score=-0.7142->-0.3571 stab_ratio=-0.01 penalty=0.50
[CPCV-PENALTY] val_score=-0.3571->-0.3323 cpcv_stab=0.50 penalty=0.93
[STABILITY-PENALTY] val_score=-1.3318->-0.6659 stab_ratio=-0.04 penalty=0.50
[18:42:02] ✓ New elite added (score=-0.6033, total=32)                                                                                                                                         
[18:42:14] ✓ New elite added (score=-0.0763, total=33)                                                                                                                                         
[18:42:14] ✓ New elite added (score=-0.0763, total=33)                                                                                                                                         
[18:42:28] ✓ New elite added (score=-0.0173, total=34)                                                                                                                                         
[18:42:28] ↗ Best score improved: -0.0599 → -0.0173                                                                                                                                            
[18:42:28] ✓ New elite added (score=0.1867, total=35)                                                                                                                                          
[18:42:28] ↗ Best score improved: -0.0173 → 0.1867                                                                                                                                             
[18:42:40] ✓ New elite added (score=0.2467, total=36)                                                                                                                                          
[18:42:40] ↗ Best score improved: 0.1867 → 0.2467                                                                                                                                              
[18:42:40] ✓ New elite added (score=-0.1055, total=37)                                                                                                                                         
[18:43:08] ✓ New elite added (score=-0.3323, total=38)                                                                                                                                         
[18:43:08] ✓ New elite added (score=-0.6196, total=39)                                                                                                                                         
[STABILITY-PENALTY] val_score=-1.8688->-0.9344 stab_ratio=-0.04 penalty=0.50
[STABILITY-PENALTY] val_score=-1.8172->-0.9086 stab_ratio=-0.04 penalty=0.50
[STABILITY-PENALTY] val_score=-2.0994->-1.0497 stab_ratio=-0.05 penalty=0.50
[STABILITY-PENALTY] val_score=-1.5884->-0.7942 stab_ratio=-0.03 penalty=0.50
[STABILITY-PENALTY] val_score=-1.8400->-0.9200 stab_ratio=-0.04 penalty=0.50
[STABILITY-PENALTY] val_score=-1.5017->-0.7508 stab_ratio=-0.03 penalty=0.50
[STABILITY-PENALTY] val_score=-1.9928->-0.9964 stab_ratio=-0.04 penalty=0.50
[STABILITY-PENALTY] val_score=-1.2083->-0.6042 stab_ratio=-0.02 penalty=0.50
[STABILITY-PENALTY] val_score=-0.9139->-0.4569 stab_ratio=-0.03 penalty=0.50
[STABILITY-PENALTY] val_score=-0.5323->-0.2662 stab_ratio=-0.01 penalty=0.50
[STABILITY-PENALTY] val_score=-1.9159->-0.9579 stab_ratio=-0.04 penalty=0.50
[STABILITY-PENALTY] val_score=-2.0283->-1.0142 stab_ratio=-0.03 penalty=0.50
[QT-VALIDATE] qtStart=50453 qtEnd=55709 warmup=220 seriesLen=72832
[QT-VALIDATE] qtBars=5256 qtWarmStart=50233 barsInSlice=5476
[QT] window: bars[50453:55709] (5256 bars) warmup_start=50233 ts=1685145600000..1704063600000 warmup=220
[QuickTest] result: trds=20 ret=-0.0689 pf=0.78 dd=0.164 fee=20 slip=5
[STABILITY-PENALTY] val_score=0.4839->0.2535 stab_ratio=0.01 penalty=0.52
[QT-VALIDATE] qtStart=50453 qtEnd=55709 warmup=221 seriesLen=72832
[QT-VALIDATE] qtBars=5256 qtWarmStart=50232 barsInSlice=5477
[QT] window: bars[50453:55709] (5256 bars) warmup_start=50232 ts=1685145600000..1704063600000 warmup=221
[QuickTest] result: trds=21 ret=-0.0548 pf=0.85 dd=0.194 fee=20 slip=5
[STABILITY-PENALTY] val_score=0.6165->0.3197 stab_ratio=0.01 penalty=0.52
[CPCV-PENALTY] val_score=0.3197->0.2975 cpcv_stab=0.50 penalty=0.93
 [META: radicalP 0.61->0.52, surExploreP 0.65->0.58]16:43:08Z  [PROG]  Batch 16: Tested 152064 | Train: 33.0817 | batchValScore: 0.6165 | Ret: 2.92% | WR: 45.6% | Trds: 206 | Rate: 710/s | Runtime: 3m10s | fp: (AND (BreakDown F[24] F[25]) (Falling F[13] 9))||(SlopeGT F[31] 0.00 8)|ATR*6.05|ATR*14.66|ATR*4.82|hold=528|loss=20|cd=150|fees=20.0000|slip=5.0000|risk=1.0000|dir=1|vol_ATR14>SMA36*0.97| ✗
  [crit: score>-5.00, trds>=30, DD<0.41(41.0%), ret>0.0%, exp>0.0001, pf>1.00, stab>50%]
  [val: score=0.6165 dd=25.74% trds=206 reason=stab=2%]
  [QuickTest: Score=-5.7828 Ret=-5.48% DD=19.40% Trds=21]
│ Stats: passRate=0.01% │ elites=41 │              ││ Meta: radicalP=0.61 │ surExploreP=0.65 │         │
│ bestValScore=6.5747 =                            ││ stagnation=168                                   │
    Regime Filter (must be true):     (SlopeGT MinusDI(F[31]) 0.00 8)
    Entry Signal (candle t close):     (AND (BreakDown ROC10(F[24]) ROC20(F[25])) (Falling RSI21(F[13]) 9))
    Entry Execution (candle t+1 open): pending entry enters at next bar open
    Exit Signal:     (Always Active - No Filter)
  [RISK MANAGEMENT]
    StopLoss: ATR*6.05
    TakeProfit: ATR*14.66
    Trail: ATR*4.82
  [WALK-FORWARD OOS RESULTS]
    GeoAvg: 2.92% | Median: 0.41% | MinMonth: -12.50% | Months: 74 | Trades: 206 | MaxDD: 25.74%
[18:42:28] ↗ Best score improved: -0.0173 → 0.1867
[18:42:40] ✓ New elite added (score=0.2467, total=36)
[18:42:40] ↗ Best score improved: 0.1867 → 0.2467
[18:42:40] ✓ New elite added (score=-0.1055, total=37)
[18:43:08] ✓ New elite added (score=-0.3323, total=38)
[18:43:08] ✓ New elite added (score=-0.6196, total=39)
[18:43:08] ✓ New elite added (score=0.2535, total=40)
[18:43:08] ↗ Best score improved: 0.2467 → 0.2535
[18:43:08] ✓ New elite added (score=0.2975, total=41)
[18:43:08] ↗ Best score improved: 0.2535 → 0.2975
Tested: 159,360 | Rate: 768/s | Best: 0.6165 | Elites: 41 | Gen: 626
gen=213873 rej_sur=0(0.0%) rej_seen=37086/250809(14.8%) rerolled=36836(17.2%) rej_novelty=0(0.0%) sent=168424(78.7%)[SAMPLED REJECTION #160001] reason=wf_oos_rejected: Max DD exceeded: 52.21% > 35.00%, seed=8424447978295043242, trades=204, ret=1.81%, pf=1.44, exp=0.00820, dd=52.21%
[QT-VALIDATE] qtStart=50453 qtEnd=55709 warmup=221 seriesLen=72832
[QT-VALIDATE] qtBars=5256 qtWarmStart=50232 barsInSlice=5477
[QT] window: bars[50453:55709] (5256 bars) warmup_start=50232 ts=1685145600000..1704063600000 warmup=221
[QuickTest] result: trds=21 ret=0.1044 pf=1.40 dd=0.135 fee=20 slip=5
[STABILITY-PENALTY] val_score=-0.7348->-0.3674 stab_ratio=-0.02 penalty=0.50
[STABILITY-PENALTY] val_score=-0.9730->-0.4865 stab_ratio=-0.06 penalty=0.50
[STABILITY-PENALTY] val_score=-1.6324->-0.8162 stab_ratio=-0.03 penalty=0.50
[STABILITY-PENALTY] val_score=-1.8108->-0.9054 stab_ratio=-0.04 penalty=0.50
[STABILITY-PENALTY] val_score=-1.6603->-0.8302 stab_ratio=-0.03 penalty=0.50
[STABILITY-PENALTY] val_score=-2.0202->-1.0101 stab_ratio=-0.05 penalty=0.50
[STABILITY-PENALTY] val_score=-2.2321->-1.1161 stab_ratio=-0.06 penalty=0.50
[STABILITY-PENALTY] val_score=-2.4226->-1.2113 stab_ratio=-0.08 penalty=0.50
[STABILITY-PENALTY] val_score=-1.0364->-0.5182 stab_ratio=-0.03 penalty=0.50
[STABILITY-PENALTY] val_score=-1.6721->-0.8360 stab_ratio=-0.04 penalty=0.50
[STABILITY-PENALTY] val_score=-2.1559->-1.0779 stab_ratio=-0.06 penalty=0.50
[STABILITY-PENALTY] val_score=-1.6598->-0.8299 stab_ratio=-0.03 penalty=0.50
[QT-VALIDATE] qtStart=50453 qtEnd=55709 warmup=220 seriesLen=72832
[QT-VALIDATE] qtBars=5256 qtWarmStart=50233 barsInSlice=5476
[QT] window: bars[50453:55709] (5256 bars) warmup_start=50233 ts=1685145600000..1704063600000 warmup=220
[QuickTest] result: trds=21 ret=-0.0439 pf=0.89 dd=0.195 fee=20 slip=5
[STABILITY-PENALTY] val_score=-0.5685->-0.2842 stab_ratio=-0.01 penalty=0.50
[STABILITY-PENALTY] val_score=-1.9044->-0.9522 stab_ratio=-0.04 penalty=0.50
[QT-VALIDATE] qtStart=50453 qtEnd=55709 warmup=220 seriesLen=72832
[QT-VALIDATE] qtBars=5256 qtWarmStart=50233 barsInSlice=5476
[QT] window: bars[50453:55709] (5256 bars) warmup_start=50233 ts=1685145600000..1704063600000 warmup=220
[QuickTest] result: trds=20 ret=-0.0719 pf=0.77 dd=0.164 fee=20 slip=5
[STABILITY-PENALTY] val_score=0.4320->0.2253 stab_ratio=0.01 penalty=0.52
[QT-VALIDATE] qtStart=50453 qtEnd=55709 warmup=221 seriesLen=72832
[QT-VALIDATE] qtBars=5256 qtWarmStart=50232 barsInSlice=5477
[QT] window: bars[50453:55709] (5256 bars) warmup_start=50232 ts=1685145600000..1704063600000 warmup=221
[QuickTest] result: trds=21 ret=-0.0450 pf=0.89 dd=0.194 fee=20 slip=5
[STABILITY-PENALTY] val_score=0.5300->0.2734 stab_ratio=0.01 penalty=0.52
[CPCV-PENALTY] val_score=0.2734->0.2544 cpcv_stab=0.50 penalty=0.93
[18:42:40] ↗ Best score improved: 0.1867 → 0.2467                                                                                                                                              
[18:42:40] ✓ New elite added (score=-0.1055, total=37)                                                                                                                                         
[18:43:08] ✓ New elite added (score=-0.3323, total=38)                                                                                                                                         
[18:43:08] ✓ New elite added (score=-0.6196, total=39)                                                                                                                                         
[18:43:08] ✓ New elite added (score=0.2535, total=40)                                                                                                                                          
[18:43:08] ↗ Best score improved: 0.2467 → 0.2535                                                                                                                                              
[18:43:08] ✓ New elite added (score=0.2975, total=41)                                                                                                                                          
[18:43:08] ↗ Best score improved: 0.2535 → 0.2975                                                                                                                                              
[18:43:20] ✓ New elite added (score=0.2253, total=42)                                                                                                                                          
[18:43:20] ✓ New elite added (score=0.2544, total=43)                                                                                                                                          
[18:43:08] ✓ New elite added (score=-0.3323, total=38)                                                                                                                                         
[18:43:08] ✓ New elite added (score=-0.6196, total=39)                                                                                                                                         
[18:43:08] ✓ New elite added (score=0.2535, total=40)                                                                                                                                          
[18:43:08] ↗ Best score improved: 0.2467 → 0.2535                                                                                                                                              
[18:43:08] ✓ New elite added (score=0.2975, total=41)                                                                                                                                          
[18:43:08] ↗ Best score improved: 0.2535 → 0.2975                                                                                                                                              
[18:43:20] ✓ New elite added (score=0.2253, total=42)                                                                                                                                          
[18:43:20] ✓ New elite added (score=0.2544, total=43)                                                                                                                                          
[18:43:20] ✓ New elite added (score=0.2331, total=44)                                                                                                                                          
[18:43:20] ✓ New elite added (score=0.0999, total=45)                                                                                                                                          
[QuickTest] result: trds=18 ret=0.0228 pf=1.16 dd=0.105 fee=20 slip=5
16:43:20Z  [PROG]  Batch 17: Tested 162048 | Train: 33.3267 | batchValScore: 0.5300 | Ret: 2.87% | WR: 45.6% | Trds: 206 | Rate: 814/s | Runtime: 3m23s | fp: (AND (BreakDown F[24] F[25]) (Falling F[13] 9))||(SlopeGT F[31] 0.00 8)|ATR*6.05|ATR*15.60|ATR*4.82|hold=528|loss=20|cd=150|fees=20.0000|slip=5.0000|risk=1.0000|dir=1|vol_ATR14>SMA36*0.97| ✗
  [crit: score>-5.00, trds>=30, DD<0.41(41.0%), ret>0.0%, exp>0.0001, pf>1.00, stab>50%]
  [val: score=0.5300 dd=25.74% trds=206 reason=stab=2%]
  [QuickTest: Score=-5.4572 Ret=-4.50% DD=19.40% Trds=21]
│ Stats: passRate=0.02% │ elites=47 │              ││ Meta: radicalP=0.61 │ surExploreP=0.65 │         │
│ bestValScore=6.5747 =                            ││ stagnation=169                                   │
    Regime Filter (must be true):     (SlopeGT MinusDI(F[31]) 0.00 8)
    Entry Signal (candle t close):     (AND (BreakDown ROC10(F[24]) ROC20(F[25])) (Falling RSI21(F[13]) 9))
    Entry Execution (candle t+1 open): pending entry enters at next bar open
    Exit Signal:     (Always Active - No Filter)
  [RISK MANAGEMENT]
    StopLoss: ATR*6.05
    TakeProfit: ATR*15.60
    Trail: ATR*4.82
  [WALK-FORWARD OOS RESULTS]
    GeoAvg: 2.87% | Median: 0.41% | MinMonth: -12.50% | Months: 74 | Trades: 206 | MaxDD: 25.74%
[18:43:08] ✓ New elite added (score=0.2535, total=40)
[18:43:08] ↗ Best score improved: 0.2467 → 0.2535
[18:43:08] ✓ New elite added (score=0.2975, total=41)
[18:43:08] ↗ Best score improved: 0.2535 → 0.2975
[18:43:20] ✓ New elite added (score=0.2253, total=42)
[18:43:20] ✓ New elite added (score=0.2544, total=43)
[18:43:20] ✓ New elite added (score=0.2331, total=44)
[18:43:20] ✓ New elite added (score=0.0999, total=45)
[18:43:20] ✓ New elite added (score=0.1133, total=46)
[18:43:20] ✓ New elite added (score=0.2110, total=47)
[MIN-MONTH-REJECT] absolute_floor: MinMonth=-20.38% < Floor=-20.00% (trds=4, config_min=-20.00%)
Tested: 166,912 | Rate: 755/s | Best: 0.6165 | Elites: 47 | Gen: 627
gen=223327 rej_sur=0(0.0%) rej_seen=39341/262502(15.0%) rerolled=39075(17.5%) rej_novelty=0(0.0%) sent=175830(78.7%)Gen: 223,327 (heavy_mut))
[MIN-MONTH-REJECT] absolute_floor: MinMonth=-24.49% < Floor=-20.00% (trds=3, config_min=-20.00%)


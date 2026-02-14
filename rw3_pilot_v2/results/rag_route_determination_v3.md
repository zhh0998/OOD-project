# S2 Kill-Switch Determination (v3, Leak-Fixed)

```
==================================================
S2 KILL-SWITCH DETERMINATION (v3, leak-fixed)
==================================================

Condition 1: whitened near Cohen's d > 0.8
  clinc150_all_MiniLM_L6_v2: d=1.71 (k*=1) -> YES
  clinc150_bge_base_en_v1.5: d=2.02 (k*=1) -> YES
  banking77_all_MiniLM_L6_v2: d=1.05 (k*=20) -> YES
  banking77_bge_base_en_v1.5: d=1.14 (k*=20) -> YES
  Pass: 4/4 (need>=2) -> YES

Condition 2: CP near det improvement >= 10pp (alpha=0.10)
  clinc150_all_MiniLM_L6_v2: +5.6pp -> NO
  clinc150_bge_base_en_v1.5: +30.8pp -> YES
  banking77_all_MiniLM_L6_v2: +8.5pp -> NO
  banking77_bge_base_en_v1.5: -1.5pp -> NO
  Pass: 1/4 (need>=2) -> NO

Condition 3: graph feature increment >= 2% (unsupervised)
  clinc150_all_MiniLM_L6_v2: +0.8% -> NO
  clinc150_bge_base_en_v1.5: +3.7% -> YES
  banking77_all_MiniLM_L6_v2: +4.7% -> YES
  banking77_bge_base_en_v1.5: +5.1% -> YES
  Pass: 3/4 (need>=2) -> YES

Condition 4: cross-dataset + cross-model robustness
  Cond1 on both datasets? True
  Cond1 on both models? True
  -> YES

Banking77 robustness: alphabetical d=1.14, random d=0.84, diff=26.2% (NOT ROBUST - depends on class partition)

==================================================
S2(v3) = YES (3/4 conditions, need>=3)

v2 -> v3 changes:
  Cond1: YES -> YES
  Cond2: YES -> NO
  Cond3: YES -> YES
  Cond4: YES -> YES
==================================================
```
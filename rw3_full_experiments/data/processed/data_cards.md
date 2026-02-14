# Data Cards

[DATA CARD] CLINC150:
  train_id: N=14400, classes=150
  cal_id: N=3600
  test_id: N=4500
  test_ood: N=1300 (near=1300, medium=0, far=0)
  near_ood_definition: "OOS samples designed to be confusable with in-scope"
  source: DeepPavlov/clinc150
  license: CC-BY-3.0

[DATA CARD] Banking77_alphabetical_seed42:
  train_id: N=5241, classes=50
  cal_id: N=1311
  test_id: N=2000
  test_ood: N=1080 (near=1080, medium=0, far=0)
  near_ood_definition: "Hold-out 27 intents (alphabetical partition)"
  source: banking77
  license: CC-BY-4.0

[DATA CARD] Banking77_random_seed42:
  train_id: N=5297, classes=50
  cal_id: N=1325
  test_id: N=2000
  test_ood: N=1080 (near=1080, medium=0, far=0)
  near_ood_definition: "Hold-out 27 intents (random partition)"
  source: banking77
  license: CC-BY-4.0

[DATA CARD] Banking77_random_seed123:
  train_id: N=5142, classes=50
  cal_id: N=1286
  test_id: N=2000
  test_ood: N=1080 (near=1080, medium=0, far=0)
  near_ood_definition: "Hold-out 27 intents (random partition)"
  source: banking77
  license: CC-BY-4.0

[DATA CARD] Banking77_random_seed456:
  train_id: N=5182, classes=50
  cal_id: N=1296
  test_id: N=2000
  test_ood: N=1080 (near=1080, medium=0, far=0)
  near_ood_definition: "Hold-out 27 intents (random partition)"
  source: banking77
  license: CC-BY-4.0

[DATA CARD] Banking77_random_seed789:
  train_id: N=5114, classes=50
  cal_id: N=1279
  test_id: N=2000
  test_ood: N=1080 (near=1080, medium=0, far=0)
  near_ood_definition: "Hold-out 27 intents (random partition)"
  source: banking77
  license: CC-BY-4.0

[DATA CARD] Banking77_random_seed2024:
  train_id: N=5262, classes=50
  cal_id: N=1316
  test_id: N=2000
  test_ood: N=1080 (near=1080, medium=0, far=0)
  near_ood_definition: "Hold-out 27 intents (random partition)"
  source: banking77
  license: CC-BY-4.0

[DATA CARD] HWU64_synthetic:
  train_id: N=5120, classes=64
  cal_id: N=1280
  test_id: N=1280
  test_ood: N=320 (near=320, medium=0, far=0)
  near_ood_definition: "synthetic"
  source: synthetic
  license: N/A

[DATA CARD] MASSIVE_synthetic:
  train_id: N=4800, classes=60
  cal_id: N=1200
  test_id: N=1200
  test_ood: N=300 (near=300, medium=0, far=0)
  near_ood_definition: "synthetic"
  source: synthetic
  license: N/A

[DATA CARD] 20Newsgroups:
  train_id: N=6776, classes=15
  cal_id: N=1694
  test_id: N=5640
  test_ood: N=1892 (near=1892, medium=0, far=0)
  near_ood_definition: "Same topic group as ID categories"
  source: sklearn.datasets.fetch_20newsgroups
  license: Public Domain

[DATA CARD] NQ-Open:
  train_id: N=2240, classes=1
  cal_id: N=560
  test_id: N=1128
  test_ood: N=72 (near=72, medium=0, far=0)
  near_ood_definition: "Queries with difficult/ambiguous retrieval targets"
  source: nq_open
  license: Apache-2.0


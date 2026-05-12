Default current: weights/best_tsm_topdown_hn1.pth
Fallback 1: weights/best_tsm_topdown_retrain_dataviolence.pth
Fallback 2: weights/best_tsm_model.pth

Pipeline commands:
1. python src/buoc6_main_pipeline.py --tsm-weights weights/best_tsm_topdown_hn1.pth
2. python src/buoc6_main_pipeline.py --tsm-weights weights/best_tsm_topdown_retrain_dataviolence.pth
3. python src/buoc6_main_pipeline.py --tsm-weights weights/best_tsm_model.pth

@echo off
echo ========================================================
echo BAT DAU HUAN LUYEN TSM MODULE (15 EPOCHS)
echo ========================================================
python src\train_tsm_topdown.py --batch-size 4 --learning-rate 1e-4 --hard-negative-list src\hard_negatives.txt --hard-negative-repeat 3
if %ERRORLEVEL% NEQ 0 (
    echo Huan luyen that bai!
    pause
    exit /b %ERRORLEVEL%
)

echo ========================================================
echo HUAN LUYEN XONG! TIEN HANH BENCHMARK TEST...
echo ========================================================
python src\benchmark_topdown_batch.py --tsm-weights weights\best_tsm_topdown.pth

echo ========================================================
echo HOAN THANH! Ban hay doc bao cao o tren hoac mo file benchmark_topdown_report.json
echo ========================================================
pause

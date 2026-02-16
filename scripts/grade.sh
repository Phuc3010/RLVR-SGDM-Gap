steps=(512)
bszs=(32 64 128)
datasets=("polaris")
# benchmarks=("Olympiad-Bench" "Minerva" "AIME24" "AIME25")
benchmarks=("Math-500")

for bench in "${benchmarks[@]}"; do
for ds in "${datasets[@]}"; do
for bsz in "${bszs[@]}"; do
for step in "${steps[@]}"; do
        python eval/grade.py --step ${step} --model_name full_${ds}_bsz${bsz}_step${step}_adamw --benchmark ${bench}

done
done
done
done
steps=(64 128 192 256 320 384 448 512)
# datasets=("math" "deepmath" "polaris")
datasets=("math" "deepmath")
benchmarks=("Olympiad-Bench" "Minerva" "AIME24" "AIME25")

for bench in "${benchmarks[@]}"; do
for ds in "${datasets[@]}"; do
for step in "${steps[@]}"; do
        python eval/grade.py --step ${step} --model_name full_${ds}_temp1.0_bsz1024_adamw --benchmark ${bench}
done
done
done
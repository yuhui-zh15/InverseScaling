# get current time from bash
curtime=$(date +"%m_%d_%H_%M")
echo $curtime

cp test.jsonl /pasteur/u/yuhuiz/archive/crfm_benchmarking/InverseScaling/eval_pipeline/helm/benchmark_output/scenarios/commonsense/data/openbookqa/OpenBookQA-V1-Sep2018/Data/Main/test.jsonl
cd /pasteur/u/yuhuiz/archive/crfm_benchmarking/InverseScaling/eval_pipeline/helm
venv/bin/benchmark-run -r commonsense:model=openai/ada,dataset=openbookqa,method=multiple_choice_joint --max-eval-instances 1000 --suite output
venv/bin/benchmark-run -r commonsense:model=openai/babbage,dataset=openbookqa,method=multiple_choice_joint --max-eval-instances 1000 --suite output
venv/bin/benchmark-run -r commonsense:model=openai/curie,dataset=openbookqa,method=multiple_choice_joint --max-eval-instances 1000 --suite output
venv/bin/benchmark-run -r commonsense:model=openai/davinci,dataset=openbookqa,method=multiple_choice_joint --max-eval-instances 1000 --suite output
cp -r /pasteur/u/yuhuiz/archive/crfm_benchmarking/InverseScaling/eval_pipeline/helm/benchmark_output/runs/output /pasteur/u/yuhuiz/archive/crfm_benchmarking/InverseScaling/eval_pipeline/results/$curtime
cp /pasteur/u/yuhuiz/archive/crfm_benchmarking/InverseScaling/eval_pipeline/helm/benchmark_output/scenarios/commonsense/data/openbookqa/OpenBookQA-V1-Sep2018/Data/Main/test.jsonl /pasteur/u/yuhuiz/archive/crfm_benchmarking/InverseScaling/eval_pipeline/results/$curtime/inputs.jsonl
cd /pasteur/u/yuhuiz/archive/crfm_benchmarking/InverseScaling/eval_pipeline/
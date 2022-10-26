# initial setup (only run once)
# git clone git@github.com:stanford-crfm/helm.git
# cd helm
# ./pre-commit-venv.sh
# echo "kGBjKfZfENL1FipevJbX6Jr3IUuPMca1" > proxy_api_key.txt
# venv/bin/benchmark-run -r commonsense:model=openai/ada,dataset=openbookqa,method=multiple_choice_joint --max-eval-instances 1000 --suite output --num-threads 8
# cd ..

# get current time from bash
curtime=$(date +"%m_%d_%H_%M")
echo $curtime

cp test.jsonl helm/benchmark_output/scenarios/commonsense/data/openbookqa/OpenBookQA-V1-Sep2018/Data/Main/test.jsonl
cd helm
venv/bin/benchmark-run -r commonsense:model=openai/ada,dataset=openbookqa,method=multiple_choice_joint --max-eval-instances 10000 --suite output --num-threads 8
venv/bin/benchmark-run -r commonsense:model=openai/babbage,dataset=openbookqa,method=multiple_choice_joint --max-eval-instances 10000 --suite output --num-threads 8
venv/bin/benchmark-run -r commonsense:model=openai/curie,dataset=openbookqa,method=multiple_choice_joint --max-eval-instances 10000 --suite output --num-threads 8
venv/bin/benchmark-run -r commonsense:model=openai/davinci,dataset=openbookqa,method=multiple_choice_joint --max-eval-instances 10000 --suite output --num-threads 8
cd ..
mv helm/benchmark_output/runs/output results/$curtime
mv helm/benchmark_output/scenarios/commonsense/data/openbookqa/OpenBookQA-V1-Sep2018/Data/Main/test.jsonl results/$curtime/inputs.jsonl

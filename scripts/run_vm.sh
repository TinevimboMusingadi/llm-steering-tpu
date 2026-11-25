#!/usr/bin/env bash
# Run the must-ship experiments on a TPU VM, in order. Each step logs to ~/lab/logs/<step>.log; a failed step
# is recorded and the next one still runs. Usage:  nohup bash scripts/run_vm.sh [steps...] > ~/lab/logs/run.log 2>&1 &
cd ~/lab
. ~/venv/bin/activate
export HF_HUB_DISABLE_XET=1 PYTHONUNBUFFERED=1
mkdir -p logs results
STEPS=${@:-m0 m1loop m1batched m2strict m2defer e1}
for s in $STEPS; do
  case $s in
    m0)        cmd="python experiments/m0_hello.py --tokens 48" ;;
    m1loop)    cmd="python experiments/m1_repro_sad.py --mode loop --no-layer-sweep" ;;
    m1batched) cmd="python experiments/m1_repro_sad.py --mode batched --batch 256" ;;
    m2strict)  cmd="python experiments/m2_batched_vs_loop.py --batch 256 --eager strict" ;;
    m2defer)   cmd="python experiments/m2_batched_vs_loop.py --batch 256 --eager defer --tag defer" ;;
    e1)        cmd="python experiments/e1_layer_grid.py --dtype float32 --eager strict --batch 256" ;;
    e1test)    cmd="python experiments/e1_layer_grid.py --topic-set test --layers 8,12,14,16,17,18,20 --strengths 1,1.5,2,3 --tag test" ;;
    e2)        cmd="python experiments/e2_tomato_map.py --batch 256" ;;
    e3)        cmd="python experiments/e3_removal_search.py --batch 256" ;;
    benchB)    cmd="python experiments/bench_abc.py --eager strict" ;;
    benchC)    cmd="python experiments/bench_abc.py --eager defer" ;;
    e4)        cmd="python experiments/e4_refusal.py --batch 256" ;;
    e5)        cmd="python experiments/e5_mix.py --batch 256" ;;
    benchD)    cmd="env TORCH_LOGS=recompiles python experiments/bench_abc.py --compile" ;;
    *) echo "unknown step $s"; continue ;;
  esac
  echo "STEP_START $s $(date +%T) :: $cmd"
  start=$(date +%s)
  if $cmd > logs/$s.log 2>&1; then status=OK; else status=FAIL; fi
  echo "STEP_END $s $status $(( $(date +%s) - start ))s"
  grep -E "^\[(m0|m1|m2|e1|e2|e3|e4|e5|bench)\]" logs/$s.log | tail -8
  [ $status = FAIL ] && tail -15 logs/$s.log
done
echo ALL_DONE

set +e
cd /c/Users/hasto/EduLLM1
run(){ out=$1; shift; echo "=== TRAIN $out ==="; mkdir -p "$out"; \
  python -m src.train --train data/dbe_fit.jsonl --out "$out" "$@" > "$out/train.log" 2>&1; \
  if [ -f "$out/adapter_model.safetensors" ]; then echo "$out OK"; else echo "$out FAILED"; tail -6 "$out/train.log"; fi; }
run out/sweep/c1_1p5b_r16_e5   --base Qwen/Qwen2.5-1.5B-Instruct --lora-r 16 --epochs 5  --lr 1e-4 --batch 2 --grad-accum 8
run out/sweep/c2_1p5b_r32_e12  --base Qwen/Qwen2.5-1.5B-Instruct --lora-r 32 --epochs 12 --lr 1e-4 --batch 2 --grad-accum 8
run out/sweep/c3_1p5b_r32_lr2  --base Qwen/Qwen2.5-1.5B-Instruct --lora-r 32 --epochs 12 --lr 2e-4 --batch 2 --grad-accum 8
run out/sweep/c4_3b_r16_e8     --base Qwen/Qwen2.5-3B-Instruct   --lora-r 16 --epochs 8  --lr 1e-4 --batch 1 --grad-accum 16
echo "SWEEP DONE"



# supervised fine-tuning
ACCELERATE_LOG_LEVEL=info \
accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3_slurm.yaml \
--num_processes=4 \
scripts/run_sft.py recipes/zephyr-7b-beta/sft/selective.yaml \
--gradient_accumulation_steps=2 \
--output_dir=./models/sft-selective \
--learning_rate=5e-7 \
--other_data_config=recipes/zephyr-7b-beta/data/mix_sft.yaml \
--use_selective=True


# BFPO
ACCELERATE_LOG_LEVEL=info \
accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3_slurm.yaml \
--num_processes=4 \
scripts/run_bfpo.py recipes/zephyr-7b-beta/bfpo/selective.yaml \
--per_device_train_batch_size=2 \
--gradient_accumulation_steps=4 \
--output_dir=./models/bfpo-selective \
--model_name_or_path=./models/sft-selective \
--other_data_config=recipes/zephyr-7b-beta/data/balance_safety_dpo.yaml \
--hub_model_id=dpo-selective-buffer-spo-shift \
--use_selective=True \
--remove_unused_columns=False


# Evaluation
python scripts/evaluate.py model@_global_=causal tasks=truthfulqa_gen,truthfulqa_mc2,crows_pairs_english,ethics_cm,ethics_justice,ethics_deontology,ethics_utilitarianism,ethics_virtue,toxigen,winogrande,bigbench_hhh_alignment_multiple_choice,bigbench_fact_checker_multiple_choice,bigbench_moral_permissibility_multiple_choice,bigbench_bbq_lite_json_multiple_choice,bigbench_known_unknowns_multiple_choice,bigbench_simple_ethical_questions_multiple_choice,realtoxicityprompts_challenge,hhh_alignment model_name=./models/bfpo-selective


python scripts/evaluate.py model@_global_=chat tasks=advbench,alert,alert_adversarial model_name=./models/bfpo-selective
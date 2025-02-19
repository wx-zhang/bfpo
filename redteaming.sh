
# red teaming
ACCELERATE_LOG_LEVEL=info \
accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3_slurm.yaml \
--num_processes=4 \
scripts/run_bfpo.py recipes/zephyr-7b-beta/bfpo/selective.yaml \
--per_device_train_batch_size=2 \
--gradient_accumulation_steps=4 \
--output_dir=./models/bfpo-selective-redteaming \
--model_name_or_path=HuggingFaceH4/zephyr-7b-beta \
--other_data_config=recipes/zephyr-7b-beta/data/redteaming.yaml \
--hub_model_id=bfpo-selective-redteaming \
--use_selective=True \
--save_total_limit=10

# Evaluation
python scripts/evaluate.py model@_global_=causal tasks=truthfulqa_gen,truthfulqa_mc2,crows_pairs_english,ethics_cm,ethics_justice,ethics_deontology,ethics_utilitarianism,ethics_virtue,toxigen,winogrande,bigbench_hhh_alignment_multiple_choice,bigbench_fact_checker_multiple_choice,bigbench_moral_permissibility_multiple_choice,bigbench_bbq_lite_json_multiple_choice,bigbench_known_unknowns_multiple_choice,bigbench_simple_ethical_questions_multiple_choice,realtoxicityprompts_challenge,hhh_alignment model_name=./models/bfpo-selective-redteaming


python scripts/evaluate.py model@_global_=chat tasks=advbench,alert,alert_adversarial model_name=./models/bfpo-selective-redteaming
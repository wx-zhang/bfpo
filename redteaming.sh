
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



# supervised fine-tuning
ACCELERATE_LOG_LEVEL=info \
accelerate launch  \
--config_file recipes/accelerate_configs/deepspeed_zero3.yaml \
--num_processes=4 \
scripts/run_sft.py recipes/zephyr-7b-beta/sft/selective.yaml \
--gradient_accumulation_steps=2 \
--output_dir=./models/sft-selective \
--learning_rate=5e-7 \
--other_data_config=recipes/zephyr-7b-beta/data/mix_sft.yaml \
--use_selective=True


# BFPO
ACCELERATE_LOG_LEVEL=info \
accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml \
--num_processes=4 \
scripts/run_bfpo.py recipes/zephyr-7b-beta/bfpo/selective.yaml \
--per_device_train_batch_size=2 \
--gradient_accumulation_steps=4 \
--output_dir=./models/bfpo-selective \
--model_name_or_path=./models/sft-selective \
--other_data_config=recipes/zephyr-7b-beta/data/balance_bfpo.yaml \
--hub_model_id=dpo-selective-buffer-spo-shift \
--use_selective=True \
--remove_unused_columns=False



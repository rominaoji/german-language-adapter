## Training a Language Adapter

You can train a language adapter using the scripts provided in the [Adapter-Hub Hugging Face repository](https://github.com/adapter-hub/adapters).

- The training script for language modeling is available [here](https://github.com/adapter-hub/adapters/blob/main/examples/pytorch/language-modeling/run_mlm.py).  
  This script is used for training a language adapter for the **mDeBERTa** model.  

### Example: Training `rominaoji/icelandic_lora8`
- The example language adapter configuration for `rominaoji/icelandic_lora8` uses LoRA (Low-Rank Adaptation).  
- You can find the LoRA configuration file here: [lora_config.json](https://github.com/rominaoji/german-language-adapter/blob/main/training/lora_config.json).  

# Weighted Preference Optimization (WPO)
This repository contains the code and released models for our paper "WPO: Enhancing RLHF with Weighted Preference Optimization". We propose a novel strategy to enhance off-policy preference optimization by simulating on-policy learning with off-policy preference data. Our Weighted Preference Optimization (WPO) method adapts off-policy data to resemble on-policy data more closely by reweighting preference pairs according to their probability under the current policy. This method not only addresses the distributional gap problem but also enhances the optimization process without incurring additional costs. WPO not only outperforms Direct Preference Optimization (DPO) by up to 5.6\% on Alpaca Eval 2 but also establishes a remarkable length-controlled winning rate against GPT-4-turbo of 48.6\% based on Llama-3-8B-Instruct, making it the strongest 8B model on the leaderboard to date.

<img src="./figures/wpo.png" width="950px"></img>

## Release
- **[7/24]** We released the training code and our trained models.
- **[6/17]** We released our preprint. We are still awaiting internal approval for releasing the code and models. Stay tuned for updates.

## Training
Our codebase is built upon the [alignment-handbook repo](https://github.com/huggingface/alignment-handbook) and [princeton-nlp/SimPO](https://github.com/princeton-nlp/SimPO).

### Environment Setup
To set up the environment, follow the installation instructions provided in the [SimPO repository](https://github.com/princeton-nlp/SimPO).

### Training Scripts

We provide training config files for training off-policy models in the paper. The training config is set for 8xH100 GPUs. You may need to adjust `per_device_train_batch_size` based on your computation environment. 

* Mistral-Base:
```shell
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file accelerate_configs/deepspeed_zero3.yaml scripts/run_wpo.py training_configs/mistral.yaml
```
* Llama3-Instruct:
```shell
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file accelerate_configs/deepspeed_zero3.yaml scripts/run_wpo.py training_configs/llama3_instruct.yaml
```

## Models
### Zephyr/Llama models
The table below presents our trained models in the paper along with their evaluation results. Please note that these results differ from those reported in the paper, as the paper provides average results, whereas the results below pertain to individual checkpoints. The v1 models are those reported in our paper. The v2 model is trained with an enhanced method for constructing preference data. We implemented two key changes:

1. **Rejecting Responses**: Instead of selecting a random response as the rejected one (as originally described in the paper), we now use the response with the minimum score.
2. **Handling Ties**: We developed a better strategy for resolving ties among responses. When multiple responses have the highest score, we select the one with the shortest length. Similarly, when multiple responses have the lowest score, we choose the one with the smallest length difference compared to the chosen output. This approach helps mitigate length bias in preference optimization.

| Checkpoint | Alpaca Eval LC | Alpaca Eval WR |
|---|---|---|
|[zephyr-7B-WPO-FP](https://huggingface.co/wzhouad/zephyr-7B-WPO-FP)|25.4|21.0|
|[zephyr-7B-WPO-HB](https://huggingface.co/wzhouad/zephyr-7B-WPO-HB)|42.9|49.8|
|[Llama3-Instruct-8B-WPO-FP](https://huggingface.co/wzhouad/Llama3-Instruct-8B-WPO-FP)|33.8|31.5|
|[Llama3-Instruct-8B-WPO-HB](https://huggingface.co/wzhouad/Llama3-Instruct-8B-WPO-HB)|48.3|52.3|
|[Llama3-Instruct-8B-WPO-HB-v2](https://huggingface.co/wzhouad/Llama3-Instruct-8B-WPO-HB-v2)|53.4|57.3|

### Gemma models
The table below shows our finetuned gemma-2-it models. When constructing the preference dataset for Gemma models in the hybrid RL setting, we switch to [ArmoRM](https://huggingface.co/RLHFlow/ArmoRM-Llama3-8B-v0.1) for scoring, and choose the outputs with maximum/minimum scores to form a preference pair.

| Checkpoint | Alpaca Eval LC | Alpaca Eval WR |
|---|---|---|
|[gemma-2-9b-it-WPO-FP](https://huggingface.co/wzhouad/gemma-2-9b-it-WPO-FP)|56.0|47.2|
|[gemma-2-9b-it-WPO-HB](https://huggingface.co/wzhouad/gemma-2-9b-it-WPO-HB)|76.7|77.8|

## Citation

Please kindly cite the following paper if you use our method or models in your work:
```bibtex
@article{zhou2024wpo,
  title={WPO: Enhancing RLHF with Weighted Preference Optimization},
  author={Zhou, Wenxuan and Agrawal, Ravi and Zhang, Shujian and Indurthi, Sathish Reddy and Zhao, Sanqiang and Song, Kaiqiang and Xu, Silei and Zhu, Chenguang},
  journal={arXiv preprint arXiv:2406.11827},
  year={2024}
}
```

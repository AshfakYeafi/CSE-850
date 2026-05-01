# Adversarial ML on STL-10

This repository contains my CSE 850 final project on adversarial robustness using the STL-10 image classification dataset. The project compares standard clean training with several adversarial training and evaluation methods, including FGSM, PGD, and TRADES.

The implementation uses a ResNet-18 backbone and includes scripts for training, evaluation, result export, ablation studies, and visualization of adversarial examples.

**Author:** Ashfak Yeafi  
**Program:** Biosystems and Agricultural Engineering, Michigan State University  
**Email:** <yeafiash@msu.edu>  
**Repository:** <https://github.com/AshfakYeafi/CSE-850>  

---

## Project overview

Deep neural networks often perform very well on clean image classification tasks, but their predictions can become unstable when small adversarial perturbations are added to the input image. In this project, I study this problem using the STL-10 dataset and evaluate how different training strategies affect clean accuracy and adversarial robustness.

The main experiments include:

- Clean training on STL-10
- FGSM adversarial training
- PGD adversarial training
- TRADES adversarial training
- Clean, FGSM, and PGD evaluation
- Transfer-attack analysis
- Per-class robustness analysis
- Robustness analysis under reduced input resolution
- PGD-step ablation study
- Visualization of adversarial examples

The goal is not only to compare accuracy values, but also to understand the trade-off between clean performance and robust performance under different adversarial settings.

---

## Repository structure

```text
.
├── main.py
├── run_ablation.py
├── requirements.txt
├── src/
│   ├── config.py
│   ├── data.py
│   ├── models.py
│   ├── train.py
│   ├── attacks.py
│   ├── evaluate.py
│   └── utils.py
└── outputs/
    ├── models/
    ├── results/
    └── plots/
```

The exact file names inside `src/` may vary depending on the latest version of the code, but the main experiments are controlled through `main.py`.

---

## Environment setup

This project was developed using a conda environment with GPU-enabled PyTorch. I used the following environment name:

```bash
conda activate torch_env
```

After activating the environment, install the required packages:

```bash
pip install -r requirements.txt
```

To check whether PyTorch can access the GPU, run:

```bash
python -c "import torch; print('cuda_available=', torch.cuda.is_available()); print('device_count=', torch.cuda.device_count()); print('gpu_name=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

If CUDA runs out of memory during training, reduce the batch size in:

```text
src/config.py
```

The default batch sizes are selected to be safe for STL-10 training on an RTX 4060 Ti.

---

## Dataset

The experiments use the STL-10 dataset. The dataset is automatically handled by the training and evaluation scripts.

STL-10 contains natural images from 10 classes. The images are larger than CIFAR-10 images, which makes the dataset useful for testing adversarial robustness under a slightly higher-resolution setting.

---

## Clean training

To train the baseline ResNet-18 model on clean STL-10 images, run:

```bash
conda activate torch_env
python main.py --mode train
```

This trains the model using only clean images. The saved model is later used for clean and adversarial evaluation.

---

## PGD adversarial training

To train the model using PGD-based adversarial training, run:

```bash
conda activate torch_env
python main.py --mode adv_train
```

PGD adversarial training is usually stronger than FGSM training because it uses multiple attack steps during training. This makes the model more exposed to stronger adversarial examples.

---

## FGSM adversarial training

To train the model using FGSM-based adversarial training, run:

```bash
conda activate torch_env
python main.py --mode fgsm_train
```

FGSM uses a single-step perturbation. It is faster than PGD, but it is usually less strong as an adversarial training method.

---

## TRADES training

To run TRADES adversarial training, use:

```bash
conda activate torch_env
python main.py --mode trades_train
```

TRADES training saves report-ready files automatically, including training logs and plots.

Example outputs:

```text
outputs/results/trades_training_log_*.csv
outputs/plots/trades_training_curves_*.png
```

These files can be used directly in the final report to show the training behavior of the TRADES model.

---

## Model evaluation

After training the models, run:

```bash
conda activate torch_env
python main.py --mode eval
```

This evaluates the trained models under clean, FGSM, and PGD settings. The results are saved as CSV files inside:

```text
outputs/results/
```

The evaluation step is useful for comparing how each training method performs on both clean and adversarial images.

---

## Robustness vs. input resolution

One additional analysis in this project studies how adversarial robustness changes when the input image resolution is reduced. For example, STL-10 images can be downsampled from `96 × 96` to lower resolutions such as `48 × 48`, and then evaluated under PGD attack.

Run the resolution experiment using:

```bash
conda run -n torch_env python main.py --mode resolution_eval
```

The generated files are saved as:

```text
outputs/results/resolution_robustness_*.csv
outputs/plots/resolution_robustness_*.png
```

This experiment helps show whether reducing image resolution changes the clean accuracy and adversarial accuracy trade-off.

---

## Generate report figures

To generate additional figures for the final report, run:

```bash
conda run -n torch_env python main.py --mode report_viz
```

The figures are saved in:

```text
outputs/plots/report/
```

Typical generated files include:

```text
clean_vs_robust_tradeoff_*.png
transfer_heatmap_fgsm_*.png
transfer_heatmap_pgd_*.png
per_class_grouped_pgd_*.png
resolution_dual_plot_*.png
```

These plots are intended to be used in the written report or presentation.

---

## Visualize adversarial examples

To generate side-by-side adversarial example figures, run:

```bash
conda activate torch_env
python main.py --mode visualize --model clean --attack PGD --epsilon 0.031373 --num_examples 8
```

This creates visual comparisons of:

- Original image
- Adversarial image
- Perturbation map

The script saves examples where the attack successfully changes the model prediction.

You can change the model and attack type with:

```bash
--model clean|adv|fgsm|trades
--attack FGSM|PGD
```

For example:

```bash
python main.py --mode visualize --model trades --attack PGD --epsilon 0.031373 --num_examples 8
```

Generated images are saved in:

```text
outputs/plots/attack_examples/
```

---

## PGD-step ablation study

To test the effect of different PGD step counts, run:

```bash
conda activate torch_env
python run_ablation.py --steps 3 7 10
```

The ablation results are saved as:

```text
outputs/results/ablation_results.csv
```

This experiment helps check how attack strength changes when the number of PGD steps increases.

---

## Output folders

The main output folders are:

```text
outputs/models/
outputs/results/
outputs/plots/
```

Models are saved in:

```text
outputs/models/
```

CSV result files are saved in:

```text
outputs/results/
```

Plots and visualizations are saved in:

```text
outputs/plots/
```

---

## Important output files

Some important result files include:

```text
outputs/results/transfer_attack_results_*.csv
outputs/results/per_class_results_*.csv
outputs/results/resolution_robustness_*.csv
outputs/results/ablation_results.csv
```

Some important plot files include:

```text
outputs/plots/per_class_robustness.png
outputs/plots/resolution_robustness_*.png
outputs/plots/attack_examples/*.png
outputs/plots/report/*.png
```

Some outputs may include dataset-specific tags such as `_stl10`, which makes it easier to separate results from different experiments.

---

## Notes on the experiments

The project is organized so that most experiments can be run from `main.py` by changing the `--mode` argument. This makes it easier to train models, evaluate attacks, and generate figures without changing the source code.

The main comparison is between:

- Standard clean training
- FGSM adversarial training
- PGD adversarial training
- TRADES adversarial training

The extra experiments, including transfer attacks, per-class robustness, input-resolution robustness, and PGD-step ablation, are included to give a broader view of model robustness on STL-10.

---

## Reproducing the main workflow

A typical workflow is:

```bash
conda activate torch_env

python main.py --mode train
python main.py --mode fgsm_train
python main.py --mode adv_train
python main.py --mode trades_train

python main.py --mode eval
python main.py --mode resolution_eval
python main.py --mode report_viz

python run_ablation.py --steps 3 7 10
```

For adversarial example visualization:

```bash
python main.py --mode visualize --model clean --attack PGD --epsilon 0.031373 --num_examples 8
```

---

## Citation and use

This repository was prepared as part of the final project for CSE 850 at Michigan State University. The code and generated results are intended for academic coursework, experimentation, and report preparation.

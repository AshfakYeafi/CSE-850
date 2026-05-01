import argparse

import pandas as pd

from src.adv_train import train_adversarial_model
from src.config import (
    ADV_MODEL_PATH,
    ATTACK_VIZ_DIR,
    CLEAN_MODEL_PATH,
    DEVICE,
    FGSM_MODEL_PATH,
    TRADES_MODEL_PATH,
    PER_CLASS_PLOT_PATH,
    PER_CLASS_RESULT_CSV_PATH,
    PLOT_PATH,
    RESULT_CSV_PATH,
    SEED,
    TRANSFER_RESULT_CSV_PATH,
    RESOLUTION_PLOT_PATH,
    RESOLUTION_RESULT_CSV_PATH,
    EVAL_RESOLUTIONS,
    TRAIN_RESOLUTION,
    ADV_TRAIN_EPSILON,
)
from src.data_loader import get_data_loaders
from src.evaluate import (
    evaluate_model,
    evaluate_on_clean,
    evaluate_per_class,
    evaluate_transfer_matrix,
    evaluate_under_attack,
)
from src.fgsm_train import train_fgsm_adversarial_model
from src.model import build_model
from src.report_visualizations import generate_report_visualizations
from src.train import train_clean_model
from src.trades_train import train_trades_model
from src.utils import (
    ensure_directories,
    load_model,
    plot_per_class_results,
    plot_results,
    print_summary_table,
    save_results_to_csv,
    set_seed,
    plot_resolution_results,
)
from src.visualize_attacks import save_attack_visualization


def parse_args():
    parser = argparse.ArgumentParser(
        description="Adversarial ML project: clean training, adversarial training, and evaluation."
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=[
            "train",
            "adv_train",
            "fgsm_train",
            "trades_train",
            "eval",
            "visualize",
            "resolution_eval",
            "report_viz",
        ],
        help="Choose whether to train a clean model, adversarial model, or evaluate saved models.",
    )
    parser.add_argument(
        "--model",
        default="clean",
        choices=["clean", "adv", "fgsm", "trades"],
        help="Model used for visualization mode.",
    )
    parser.add_argument(
        "--attack",
        default="PGD",
        choices=["FGSM", "PGD"],
        help="Attack type for visualization mode.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=8 / 255,
        help="Attack epsilon for visualization mode.",
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=8,
        help="Number of successful adversarial examples to save.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(SEED)
    ensure_directories()
    train_loader, test_loader = get_data_loaders()

    if args.mode == "train":
        model = build_model()
        train_clean_model(model, train_loader, test_loader)
        return

    if args.mode == "adv_train":
        model = build_model()
        train_adversarial_model(model, train_loader, test_loader)
        return

    if args.mode == "fgsm_train":
        model = build_model()
        train_fgsm_adversarial_model(model, train_loader, test_loader)
        return

    if args.mode == "trades_train":
        model = build_model()
        train_trades_model(model, train_loader, test_loader)
        return

    if args.mode == "visualize":
        model_path_by_name = {
            "clean": CLEAN_MODEL_PATH,
            "adv": ADV_MODEL_PATH,
            "fgsm": FGSM_MODEL_PATH,
            "trades": TRADES_MODEL_PATH,
        }
        model_label_by_name = {
            "clean": "Clean Model",
            "adv": "Adversarial Model",
            "fgsm": "FGSM Adversarial Model",
            "trades": "TRADES Model",
        }
        selected_model = load_model(
            build_model(),
            model_path_by_name[args.model],
            DEVICE,
        )
        out_path = ATTACK_VIZ_DIR / (
            f"{args.model}_{args.attack.lower()}_eps_{args.epsilon:.6f}.png"
        )
        saved_path = save_attack_visualization(
            model=selected_model,
            data_loader=test_loader,
            attack_name=args.attack,
            epsilon=args.epsilon,
            model_name=model_label_by_name[args.model],
            output_path=out_path,
            num_examples=args.num_examples,
        )
        if saved_path is not None:
            print(f"Saved attack visualization to {saved_path}")
        return

    if args.mode == "resolution_eval":
        clean_model = load_model(build_model(), CLEAN_MODEL_PATH, DEVICE)
        adv_model = load_model(build_model(), ADV_MODEL_PATH, DEVICE)
        fgsm_model = load_model(build_model(), FGSM_MODEL_PATH, DEVICE)

        models = {
            "Clean Model": clean_model,
            "PGD-trained Model": adv_model,
            "FGSM-trained Model": fgsm_model,
        }

        rows = []
        for res in EVAL_RESOLUTIONS:
            _, test_loader_res = get_data_loaders(
                train_resolution=TRAIN_RESOLUTION,
                eval_resolution=res,
            )
            for model_name, model in models.items():
                clean_acc = evaluate_on_clean(
                    model,
                    test_loader_res,
                    desc=f"{model_name} res={res} clean",
                )
                pgd_acc = evaluate_under_attack(
                    model,
                    test_loader_res,
                    attack_name="PGD",
                    epsilon=ADV_TRAIN_EPSILON,
                    desc=f"{model_name} res={res} PGD eps={ADV_TRAIN_EPSILON:.6f}",
                )
                rows.append(
                    {
                        "model": model_name,
                        "resolution": res,
                        "clean_accuracy": clean_acc,
                        "pgd_attack_epsilon": ADV_TRAIN_EPSILON,
                        "pgd_accuracy": pgd_acc,
                    }
                )

        resolution_df = pd.DataFrame(rows)
        save_results_to_csv(resolution_df, RESOLUTION_RESULT_CSV_PATH)
        plot_resolution_results(resolution_df, RESOLUTION_PLOT_PATH)
        print(f"Saved resolution robustness results to {RESOLUTION_RESULT_CSV_PATH}")
        print(f"Saved resolution robustness plot to {RESOLUTION_PLOT_PATH}")
        return

    if args.mode == "report_viz":
        generate_report_visualizations()
        return

    clean_model = load_model(build_model(), CLEAN_MODEL_PATH, DEVICE)
    adv_model = load_model(build_model(), ADV_MODEL_PATH, DEVICE)
    fgsm_model = load_model(build_model(), FGSM_MODEL_PATH, DEVICE)
    trades_model = load_model(build_model(), TRADES_MODEL_PATH, DEVICE)

    clean_results = evaluate_model(clean_model, test_loader, "Clean Model")
    adv_results = evaluate_model(adv_model, test_loader, "Adversarial Model")
    fgsm_results = evaluate_model(fgsm_model, test_loader, "FGSM Adversarial Model")
    trades_results = evaluate_model(trades_model, test_loader, "TRADES Model")
    all_results = pd.concat(
        [clean_results, adv_results, fgsm_results, trades_results], ignore_index=True
    )

    save_results_to_csv(all_results, RESULT_CSV_PATH)
    plot_results(all_results, PLOT_PATH)
    print_summary_table(all_results)

    transfer_df = evaluate_transfer_matrix(
        {
            "Clean Model": clean_model,
            "Adversarial Model": adv_model,
            "FGSM Adversarial Model": fgsm_model,
            "TRADES Model": trades_model,
        },
        test_loader,
    )
    save_results_to_csv(transfer_df, TRANSFER_RESULT_CSV_PATH)

    per_class_df = pd.concat(
        [
            evaluate_per_class(clean_model, test_loader, "Clean Model"),
            evaluate_per_class(adv_model, test_loader, "Adversarial Model"),
            evaluate_per_class(fgsm_model, test_loader, "FGSM Adversarial Model"),
            evaluate_per_class(trades_model, test_loader, "TRADES Model"),
        ],
        ignore_index=True,
    )
    save_results_to_csv(per_class_df, PER_CLASS_RESULT_CSV_PATH)
    plot_per_class_results(per_class_df, PER_CLASS_PLOT_PATH)

    print(f"Saved results to {RESULT_CSV_PATH}")
    print(f"Saved plot to {PLOT_PATH}")
    print(f"Saved transfer results to {TRANSFER_RESULT_CSV_PATH}")
    print(f"Saved per-class results to {PER_CLASS_RESULT_CSV_PATH}")
    print(f"Saved per-class plot to {PER_CLASS_PLOT_PATH}")


if __name__ == "__main__":
    main()

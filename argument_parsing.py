import argparse


def function_calling_argument_parsing():
    parser = argparse.ArgumentParser()
    parser.add_argument("--neftune_alpha", type=float, default=None, required=False)
    parser.add_argument("--use_default_datacollator", action='store_true', required=False)
    parser.add_argument("--restore", action='store_true', required=False)
    parser.add_argument("--checkpoint_name", type=str, default='mistral_model', required=False)
    parser.add_argument("--model_path", type=str, default='/hdd_second/goncharenko', required=False)
    parser.add_argument("--total_train_steps", type=int, default=8000, required=False)

    args = parser.parse_args()
    return args




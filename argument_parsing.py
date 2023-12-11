import argparse


def argument_parsiong():
    parser = argparse.ArgumentParser()
    parser.add_argument("neftune", type=float, default=None)
    parser.add_argument("use_default_datacollator", action='store_true')
    parser.add_argument("restore", action='store_true')
    parser.add_argument("checkpoint_name", type=str, default='mistral_model')
    parser.add_argument("model_path", type=str, default='.')
    parser.add_argument("total_train_steps", type=int, default=4000)

    args = parser.parse_args()
    return args




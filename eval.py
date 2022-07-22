import os
import argparse
from PIL import Image
from tool.predictor import Predictor
from tool.config import Cfg
from tool.utils import compute_accuracy, AverageMeter
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, help='foo help')
    parser.add_argument('--txt-folder', required=True, help='foo help')
    parser.add_argument('--config', required=True, help='foo help')

    args = parser.parse_args()
    config = Cfg.load_config_from_file(args.config)

    detector = Predictor(config)
    df = open(args.txt_folder, 'r').readlines()
    df = [x.strip().split("\t") for x in df]
    acc_seq_metric = AverageMeter()
    acc_char_metric = AverageMeter()
    pdar = tqdm(df, desc="Evaluating")
    for ip, label in pdar:
        img_path = os.path.join(args.root, ip)
        img = Image.open(img_path)
        s = detector.predict(img)
        acc_seq = compute_accuracy([s], [label], "full_sequence")
        acc_char = compute_accuracy([s], [label], "per_char")
        acc_seq_metric.update(acc_seq)
        acc_char_metric.update(acc_char)
        pdar.set_postfix(acc_seq=acc_seq_metric.avg, acc_char=acc_char_metric.avg)
    print("Sequence level accuracy: {}".format(acc_seq_metric.avg))
    print("Character level accuracy: {}".format(acc_char_metric.avg))


if __name__ == '__main__':
    main()

import argparse

import torch

from vedacore.fileio import dump
from vedacore.misc import Config, ProgressBar, load_weights, DictAction
from vedacore.parallel import MMDataParallel
from vedatad.datasets import build_dataloader, build_dataset
from vedatad.engines import build_engine


def parse_args():
    parser = argparse.ArgumentParser(description='Test a detector')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')

    args = parser.parse_args()
    return args


def prepare(cfg, checkpoint):

    engine = build_engine(cfg.val_engine)
    load_weights(engine.model, checkpoint, map_location='cpu')

    device = torch.cuda.current_device()
    engine = MMDataParallel(
        engine.to(device), device_ids=[torch.cuda.current_device()])

    dataset = build_dataset(cfg.data.val, dict(test_mode=True))
    dataloader = build_dataloader(dataset, 1, 1, dist=False, shuffle=False)

    return engine, dataloader


def test(engine, data_loader):
    engine.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):

        with torch.no_grad():
            result = engine(data)[0]

        results.append(result)
        batch_size = len(data['video_metas'][0].data)
        for _ in range(batch_size):
            prog_bar.update()
    return results


def main():

    args = parse_args()
    cfg = Config.fromfile(args.config)

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    engine, data_loader = prepare(cfg, args.checkpoint)

    results = test(engine, data_loader)

    if args.out:
        print(f'\nwriting results to {args.out}')
        dump(results, args.out)

    kwargs = dict() if args.eval_options is None else args.eval_options
    data_loader.dataset.evaluate(results, **kwargs)


if __name__ == '__main__':
    main()

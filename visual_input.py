import argparse
import datetime
import deepdish as dd
import matplotlib.pyplot as plt
import numpy as np

classes = ['Normal sinus', 'Atrial fibrillation', 'Atrial flutter', 'Ventricular bigeminy', 'Ventricular trigeminy', 'Noise']

input_sizes = [256, 512, 768, 1024, 1536]

def count_peaks_before_key(peaks, key):
    count = 0
    for k, ps in peaks.items():
        if k == key:
            break
        count += len(ps)
    return count

def locate(args):
    from bisect import bisect_left
    record = args.record
    sample = args.sample
    key = 's' + str(record)
    peaks = dd.io.load(f'{args.working_dataset_path}/trainpeak.hdf5')
    dataset = 'train'
    if key not in peaks:
        peaks = dd.io.load(f'{args.working_dataset_path}/testpeak.hdf5')
        dataset = 'test'
    if key not in peaks:
        print(f'Record {record} not found in peak lists.')
        return
    count = count_peaks_before_key(peaks, key)
    peaks = peaks[key]
    i = bisect_left(peaks, sample) # peaks are supposed to be sorted
    print(f'Found in {dataset} {record} Index {i} Dataset {count + i} Sample {peaks[i]}')
    print(peaks[i - 2:i + 3])
    return dataset, count + i, peaks[i]

def test(args):
    data = []
    label = []
    record = []
    sample = []
    dataset = 'test'
    if args.record is not None:
        dataset, dataset_i, p = locate(args)
        record.append(args.record)
        sample.append(p)
    else:
        peak = dd.io.load(f'{args.working_dataset_path}/{dataset}peak.hdf5')
        for rec, peaks in peak.items():
            record.extend([int(rec[1:])] * len(peaks))
            sample.extend(peaks)
            if args.first is not None and len(record) > args.first:
                record = record[:args.first]
                sample = sample[:args.first]
                break
    for input_size in input_sizes:
        d = dd.io.load(f'{args.all_input_path}/{dataset}{input_size}.hdf5')[args.lead]
        l = dd.io.load(f'{args.all_input_path}/{dataset}label{input_size}.hdf5')[args.lead]
        if args.record is not None:
            d = d[dataset_i:dataset_i + 1, :]
            l = l[dataset_i:dataset_i + 1, :]
        elif args.first is not None:
            d = d[:args.first, :]
            l = l[:args.first, :]
        data.append(d)
        label.append(l)
    count = 0
    #np.random.seed(0)
    for i, (r, s) in enumerate(zip(record, sample)):
        fig, ax = plt.subplots()
        prev_samp_label = None
        for is_i, (input_size, d, l) in enumerate(zip(input_sizes, data, label)):
            offset = (len(input_sizes) - is_i - 1) * 5
            sample_data = d[i]
            sample_label = l[i]
            sl_i = np.argmax(sample_label)
            if prev_samp_label is not None and prev_samp_label != sl_i:
                print('Different label found in input size')
            prev_samp_label = sl_i
            ax.plot(np.arange(s - input_size // 2, s + input_size // 2),
                    sample_data + offset,
                    label=f'{input_size} (+{offset})')
        delta = datetime.timedelta(seconds=s/args.sample_rate)
        ax.set_title(f'Record {r} centred at {delta} True: {classes[sl_i]}')
        ax.set_ylabel('normalised voltage')
        ax.set_xlabel('sample number (seconds × sampling rate)')
        ax.legend()
        plt.show()
        count += 1
        if args.limit is not None and count == args.limit:
            break

def run_data(args):
    import os
    from tqdm import tqdm
    import data
    if args.physhik_path is None: # logic in data.main(), need to figure out where the output files go
        output_path = args.train_output_dataset_path
    else:
        output_path = f'{args.physhik_path}/dataset'
    data_args = data.parser.parse_args([])
    for attr in ['working_dataset_path', 'misc_path', 'physhik_path', 'dataset_path', 'train_output_dataset_path', 'cinc_base_path']:
        setattr(data_args, attr, getattr(args, attr))

    data_args.save_peaks = True
    data_args.no_save = True
    data_args.input_size = 2048
    data.main(data_args)

    data_args.save_peaks = False
    data_args.no_save = False
    data_args.load_peaks = True
    for input_size in tqdm(input_sizes):
        data_args.input_size = input_size
        data.main(data_args)
        for name in ['train', 'trainlabel', 'test', 'testlabel']:
            os.replace(f'{output_path}/{name}.hdf5', f'{args.all_input_path}/{name}{input_size}.hdf5')

def main(args):
    if args.run_data:
        run_data(args)
    if args.no_run and args.record is not None:
        locate(args)
    if not args.no_run:
        test(args)

parser = argparse.ArgumentParser(description='Predict with trained model.')
parser.add_argument('--all_input_path', default='misc',
        help='where to save output files train|test[label]input_size.hdf5 (used for extraction and visualisation)')

parser.add_argument('--working_dataset_path', default='dataset',
        help='where to store peak data and config.hdf5, these are new files added by this code')
parser.add_argument('--misc_path', default='misc',
        help='where to store other files added by this code')
parser.add_argument('--physhik_path',
        help='if set, overrides all paths below corresponding to files physhik implementation already downloads/generates')
parser.add_argument('--dataset_path', default='dataset')
parser.add_argument('--train_output_dataset_path', default='dataset',
        help='where to save output files train|test[label].hdf5')
parser.add_argument('--cinc_base_path', default='.',
        help='where to save/load cinc training2017.zip and extract to /training2017')

#parser.add_argument('--input_size', default=256)
#parser.add_argument('--model_path', default='models/MLII-latest.hdf5')
parser.add_argument('--lead', default='MLII')
parser.add_argument('--sample_rate', default=360, type=int, help='Sample rate in Hz.')

parser.add_argument('--run_data', default=False, action='store_true', help='Run data.py for each sample size and store in dataset_path')

parser.add_argument('--no_run', default=False, action='store_true')
parser.add_argument('--stats', default=False, action='store_true')
parser.add_argument('--first', default=None, type=int, help='Only use first x samples.')
parser.add_argument('--limit', default=None, type=int, help='Only show x mistakes.')

parser.add_argument('--record', default=None, type=int, help='Record to locate nearby peaks.')
parser.add_argument('--sample', default=None, type=int, help='Sample number.')

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)

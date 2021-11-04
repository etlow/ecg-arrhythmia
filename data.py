import argparse
import numpy as np
import os
from tqdm import tqdm
from wfdb import rdrecord, rdann

def get_record_names(path):
    record_names = []
    for filename in os.listdir(path):
        root, ext = os.path.splitext(filename)
        if ext == '.atr':
            record_names.append(root)
    return record_names

def get_record(path, record_name):
    record_path = os.path.join(path, record_name)
    return rdrecord(record_path), rdann(record_path, 'atr')

def get_beats(ann, ranges):
    # input: [(start, end), (start, end), ...], return: [[beat, beat], [beat], ...]
    # find all symbols (including beat classes) between start and end
    # consistent with rdann(sampfrom, sampto), sampto is inclusive
    # symbol = [symb for samp, symb in zip(ann.sample, ann.symbol) if samp >= start and samp <= end]
    # optimised implementation assuming ranges and samples are sorted
    range_symbols = []
    symbol_cache_pos = 0
    for start, end in ranges:
        symbols = []
        i = symbol_cache_pos
        while i < len(ann.sample) and ann.sample[i] <= end:
            if ann.sample[i] >= start:
                symbols.append(ann.symbol[i])
            else:
                symbol_cache_pos = i + 1 # later peaks cannot use this sample
            i += 1
        range_symbols.append(symbols)
    return range_symbols

def extract_rhythms(ann):
    rhy_i = [i for i, symb in enumerate(ann.symbol) if symb == '+']
    rhy_samples = [ann.sample[i] for i in rhy_i]
    rhy_rhythms = [ann.aux_note[i] for i in rhy_i]
    for i in range(len(rhy_rhythms)):
        if rhy_rhythms[i][-1] == '\x00':
            rhy_rhythms[i] = rhy_rhythms[i][:-1]
    return (rhy_samples, rhy_rhythms)

def get_rhythms(ann, ranges):
    # input: [(start, end), (start, end), ...], return: [[rhythm, rhythm], [rhythm], ...]
    # return all rhythms between start and end inclusive
    # optimised implementation assuming ranges and samples are sorted
    rhy_samples, rhy_rhythms = extract_rhythms(ann)
    range_rhythms = []
    rhythm_cache_pos = 0
    for start, end in ranges:
        rhythms = []
        i = rhythm_cache_pos
        while i < len(rhy_samples) and rhy_samples[i] <= start:
            i += 1
            symbol_cache_pos = i
        # now rhy_samples[i] is first rhythm after start
        if i > 0:
            rhythms.append(rhy_rhythms[i - 1])
        while i < len(rhy_samples) and rhy_samples[i] <= end:
            rhythms.append(rhy_rhythms[i])
            i += 1
        range_rhythms.append(rhythms)
    return range_rhythms

# download progress code from https://pypi.org/project/tqdm/
class TqdmUpTo(tqdm):
    """Provides `update_to(n)` which uses `tqdm.update(delta_n)`."""
    def update_to(self, b=1, bsize=1, tsize=None):
        """
        b  : int, optional
            Number of blocks transferred so far [default: 1].
        bsize  : int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize  : int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            self.total = tsize
        return self.update(b * bsize - self.n)  # also sets self.n = b * bsize

def download(args):
    from csv import reader
    from urllib.request import urlretrieve
    from zipfile import ZipFile
    # download mitdb
    base_link = 'https://physionet.org/files/mitdb/1.0.0/'
    recs = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 111, 112, 113, 114, 115, 116, 117, 118, 119, 121, 122, 123, 124,
            200, 201, 202, 203, 205, 207, 208, 209, 210, 212, 213, 214, 215, 217, 219, 220, 221, 222, 223, 228, 230, 231, 232, 233, 234]
    exts = ['atr', 'dat', 'hea']
    rec_s = set(str(rec) for rec in recs)
    os.makedirs(args.dataset_path, exist_ok=True)
    mitdb_exists = True
    for rec in recs:
        for ext in exts:
            if not os.path.isfile(f'{args.dataset_path}/{rec}.{ext}'):
                mitdb_exists = False
                break
    def extract_mitdb():
        name_base = 'mit-bih-arrhythmia-database-1.0.0/'
        with ZipFile(zip_path, 'r') as zip_ref:
            for info in zip_ref.infolist():
                rec = info.filename[-7:-4]
                ext = info.filename[-3:]
                if rec in rec_s and ext in exts and info.filename == f'{name_base}{rec}.{ext}':
                    info.filename = f'{rec}.{ext}'
                    zip_ref.extract(info, args.dataset_path)
    if not mitdb_exists:
        os.makedirs(args.misc_path, exist_ok=True)
        zip_path = f'{args.misc_path}/mit-bih-arrhythmia-database-1.0.0.zip'
        try:
            extract_mitdb()
        except:
            url = 'https://physionet.org/static/published-projects/mitdb/mit-bih-arrhythmia-database-1.0.0.zip'
            with TqdmUpTo(unit='B', unit_scale=True, unit_divisor=1024, miniters=1,
                    desc='Downloading mitdb') as t:  # all optional kwargs
                filename, _ = urlretrieve(url, zip_path, t.update_to)
            extract_mitdb()

    # download cinc
    path = args.cinc_base_path
    try:
        list(reader(open(f'{path}/training2017/REFERENCE.csv')))
        return
    except:
        pass # expected that file does not exist
    zip_path = f'{path}/training2017.zip'
    try: # extracting and see if it works
        with ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall()
        list(reader(open(f'{path}/training2017/REFERENCE.csv')))
        return
    except:
        print(f'{zip_path} does not have the right format, redownloading')
    url = 'https://archive.physionet.org/challenge/2017/training2017.zip'
    with TqdmUpTo(unit='B', unit_scale=True, unit_divisor=1024, miniters=1,
            desc='Downloading training2017.zip') as t:  # all optional kwargs
        filename, _ = urlretrieve(url, zip_path, t.update_to)
    with ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall()

# adapted from https://github.com/physhik/ecg-mit-bih
from scipy.signal import find_peaks
from sklearn import preprocessing
def get_noise(path, input_size, choose_size):
    from csv import reader
    labels = list(reader(open(f'{path}/training2017/REFERENCE.csv')))
    noises = {'trainset': [], 'testset': []}
    for i, label in enumerate(labels):
        if i > len(labels) / 6:
            set_name = 'trainset'
        else:
            set_name = 'testset'
        if label[1] == '~':
            from scipy.io import loadmat
            noise = loadmat(f'{path}/training2017/{label[0]}.mat')['val']
            noise = noise.reshape(-1)
            from scipy.signal import resample
            noise = resample(noise, int(len(noise) * 360 / 300)) # resample to match the data sampling rate 360(mit), 300(cinc)
            noise = preprocessing.scale(noise)
            noise = noise / 1000 * 6 # rough normalize, to be improved 
            peaks, _ = find_peaks(noise, distance=150)
            choices = 10 # 256*10 from 9000
            picked_peaks = np.random.choice(peaks, choices, replace=False)
            for peak in picked_peaks:
                start, end = peak - input_size // 2, peak + input_size // 2
                start_choose, end_choose = peak - choose_size // 2, peak + choose_size // 2
                if start_choose > 0 and end_choose < len(noise):
                    noises[set_name].append(noise[start:end])
    return noises

def save_physhik(args):
    import deepdish as dd
    record_names = get_record_names(args.dataset_path)
    features = ['MLII', 'V1', 'V2', 'V4', 'V5']
    testset = ['101', '105', '114', '118', '124', '201', '210', '217']
    trainset = [x for x in record_names if x not in testset]
    np.random.seed(0) # seed(0) before and after dataprocess() in physhik for hdf5 only differing in timestamps
    noises = get_noise(args.cinc_base_path, args.input_size, args.input_size)
    def saver(rec_names, set_name, sig_name, label_name):
        np.random.seed(0)
        classes = ['N','V','/','A','F','~']#,'L','R',f','j','E','a']#,'J','Q','e','S']
        datadict = {n: [] for n in features}
        datalabel = {n: [] for n in features}
        for name in tqdm(rec_names, f'Processing {set_name}'):
            record, ann = get_record(args.dataset_path, name)
            sig0 = preprocessing.scale(record.p_signal[:, 0]).tolist()
            sig1 = preprocessing.scale(record.p_signal[:, 1]).tolist()
            peaks, _ = find_peaks(sig0, distance=150)
            feature0, feature1 = record.sig_name[0], record.sig_name[1]
            ranges = [(peak - args.input_size // 2, peak + args.input_size // 2) for peak in peaks[1:-1]]
            symbols = get_beats(ann, ranges)
            for (start, end), symbol in zip(ranges, symbols):
                if len(symbol) == 1 and symbol[0] in classes and (symbol[0] != 'N' or np.random.random() < 0.15):
                    y = [0] * len(classes)
                    y[classes.index(symbol[0])] = 1
                    datalabel[feature0].append(y)
                    datalabel[feature1].append(y)
                    datadict[feature0].append(sig0[start:end])
                    datadict[feature1].append(sig1[start:end])
        for feature in ['MLII', 'V1']: 
            n = noises[set_name]
            datadict[feature] = np.concatenate((datadict[feature], n))
            noise_label = [0] * len(classes)
            noise_label[classes.index('~')] = 1
            noise_label = [noise_label] * len(n)
            datalabel[feature] = np.concatenate((datalabel[feature], noise_label))
        dd.io.save(sig_name, datadict)
        dd.io.save(label_name, datalabel)

    os.makedirs('dataset', exist_ok=True)
    saver(trainset, 'trainset', 'dataset/train.hdf5', 'dataset/trainlabel.hdf5')
    saver(testset, 'testset', 'dataset/test.hdf5', 'dataset/testlabel.hdf5')

def get_info(path):
    record_names = get_record_names(path)
    classes = ['(N', '(AFIB', '(AFL', '(B', '(T', '~']
    features = ['MLII', 'V1', 'V2', 'V4', 'V5']
    testset = ['101', '105', '114', '118', '124', '201', '202', '207', '210', '217']
    trainset = [x for x in record_names if x not in testset]
    return (classes, features, testset, trainset)

def save_peaks(args):
    import deepdish as dd
    classes, features, testset, trainset = get_info(args.dataset_path)
    def saver(rec_names, set_name, peak_filename):
        np.random.seed(0)
        record_peaks = {'s' + rec: [] for rec in rec_names}
        for name in tqdm(rec_names, f'Saving {set_name} peaks'):
            record, ann = get_record(args.dataset_path, name)
            sig0 = preprocessing.scale(record.p_signal[:, 0]).tolist()
            sig1 = preprocessing.scale(record.p_signal[:, 1]).tolist()
            peaks, _ = find_peaks(sig0, distance=150)
            peaks = [peak for peak in peaks
                    if peak - args.extract_size // 2 >= 0 and peak + args.extract_size // 2 < len(sig0)]
            ranges = [(peak - args.extract_size // 2, peak + args.extract_size // 2) for peak in peaks]
            symbols = get_rhythms(ann, ranges)
            for peak, symbol in zip(peaks, symbols):
                if len(symbol) == 1 and symbol[0] in classes and (symbol[0] != '(N' or np.random.random() < 0.15):
                    record_peaks['s' + name].append(peak)
        dd.io.save(peak_filename, record_peaks)

    w_path = args.working_dataset_path
    os.makedirs(w_path, exist_ok=True)
    saver(trainset, 'trainset', f'{w_path}/trainpeak.hdf5')
    saver(testset, 'testset', f'{w_path}/testpeak.hdf5')

    dd.io.save(f'{w_path}/config.hdf5', {'peak_choose_size': args.input_size})

def save_new(args):
    import deepdish as dd
    classes, features, testset, trainset = get_info(args.dataset_path)
    np.random.seed(0) # seed(0) before and after dataprocess() in physhik for hdf5 only differing in timestamps
    if args.load_peaks:
        noises = get_noise(args.cinc_base_path, args.input_size, dd.io.load(f'{args.working_dataset_path}/config.hdf5')['peak_choose_size'])
    else:
        noises = get_noise(args.cinc_base_path, args.input_size, args.input_size)
    def saver(rec_names, set_name, sig_name, label_name, peak_name):
        np.random.seed(0)
        datadict = {n: [] for n in features}
        datalabel = {n: [] for n in features}
        if args.load_peaks:
            record_peaks = dd.io.load(peak_name)
        for name in tqdm(rec_names, f'Processing {set_name}'):
            record, ann = get_record(args.dataset_path, name)
            sig0 = preprocessing.scale(record.p_signal[:, 0]).tolist()
            sig1 = preprocessing.scale(record.p_signal[:, 1]).tolist()
            if args.load_peaks:
                peaks = record_peaks['s' + name]
            else:
                peaks, _ = find_peaks(sig0, distance=150)
            feature0, feature1 = record.sig_name[0], record.sig_name[1]
            ranges = [(peak - args.input_size // 2, peak + args.input_size // 2) for peak in peaks
                    if peak - args.input_size // 2 >= 0 and peak + args.input_size // 2 < len(sig0)]
            symbols = get_rhythms(ann, ranges)
            for (start, end), symbol in zip(ranges, symbols):
                if len(symbol) == 1 and symbol[0] in classes:
                    if args.load_peaks or symbol[0] != '(N' or np.random.random() < 0.15: # if peaks loaded no need to filter
                        y = [0] * len(classes)
                        y[classes.index(symbol[0])] = 1
                        datalabel[feature0].append(y)
                        datalabel[feature1].append(y)
                        datadict[feature0].append(sig0[start:end])
                        datadict[feature1].append(sig1[start:end])
                elif args.load_peaks:
                    print('Loaded peak should only have 1 rhythm around it, check extract_size when loading >= current extract_size?')
                    print('or near start of record', name, start, end, symbol)
        for feature in ['MLII', 'V1']: 
            n = noises[set_name]
            datadict[feature] = np.concatenate((datadict[feature], n))
            noise_label = [0] * len(classes)
            noise_label[classes.index('~')] = 1
            noise_label = [noise_label] * len(n)
            datalabel[feature] = np.concatenate((datalabel[feature], noise_label))
        # for loop below gives smaller files but differs from original hdf5
        for feature in ['V2', 'V4', 'V5']:
            datadict[feature] = np.array(datadict[feature])
            datalabel[feature] = np.array(datalabel[feature])
        dd.io.save(sig_name, datadict)
        dd.io.save(label_name, datalabel)

    w_path = args.working_dataset_path
    t_path = args.train_output_dataset_path
    os.makedirs(w_path, exist_ok=True)
    os.makedirs(t_path, exist_ok=True)
    saver(trainset, 'trainset', f'{t_path}/train.hdf5', f'{t_path}/trainlabel.hdf5', f'{w_path}/trainpeak.hdf5')
    saver(testset, 'testset', f'{t_path}/test.hdf5', f'{t_path}/testlabel.hdf5', f'{w_path}/testpeak.hdf5')

parser = argparse.ArgumentParser(description='Extract ECG data.')
parser.add_argument('--input_size', default=256, type=int,
        help='number of samples input into model')
parser.add_argument('--extract_size', type=int,
        help='number of samples extracted from data, default input_size, (partially implemented, just use input_size)')

parser.add_argument('--working_dataset_path', default='dataset',
        help='where to store peak data and config.hdf5, these are new files added by this code')
parser.add_argument('--misc_path', default='misc',
        help='where to store other files added by this code')
parser.add_argument('--physhik_path',
        help='if set, overrides all paths below corresponding to files physhik implementation already downloads/generates')
parser.add_argument('--dataset_path', default='dataset')
parser.add_argument('--train_output_dataset_path', default='dataset',
        help='where to save output files train|test[label].hdf5 (used for training)')
parser.add_argument('--cinc_base_path', default='.',
        help='where to save/load cinc training2017.zip and extract to /training2017')

parser.add_argument('--download', default=False, action='store_true',
        help='check if required datasets exist, if not download them')
parser.add_argument('--save_physhik', default=False, action='store_true')
parser.add_argument('--no_save', default=False, action='store_true',
        help='do not actually preprocess and save the data')
parser.add_argument('--save_peaks', default=False, action='store_true')
parser.add_argument('--load_peaks', default=False, action='store_true',
        help='use saved peaks in (working_dataset_path)/*.hdf5 for preprocessing')
parser.add_argument('--check_equal', default=False, action='store_true')
parser.add_argument('--print_classes', default=False, action='store_true',
        help='print count of each class (both train and test)')
parser.add_argument('--print_rhythms', default=False, action='store_true',
        help='also print rhythms present in each record, can use with --no_save')

def main(args):
    if args.extract_size is None:
        args.extract_size = args.input_size
    if args.physhik_path is not None:
        args.dataset_path = args.physhik_path + '/dataset'
        args.train_output_dataset_path = args.physhik_path + '/dataset'
        args.cinc_base_path = args.physhik_path

    if args.download:
        download(args)

    if args.save_peaks:
        save_peaks(args)

    if not args.no_save:
        if args.save_physhik:
            save_physhik(args)
        else:
            save_new(args)

    if args.print_rhythms:
        from collections import Counter
        records = get_record_names(args.dataset_path)
        total = Counter()
        for record_name in records:
            rec, ann = get_record(args.dataset_path, record_name)
            # print(record_name, rec.p_signal.sum())
            # print(record_name, Counter(ann.aux_note))
            rhy_samples, rhy_rhythms = extract_rhythms(ann)
            rhy_end = np.append(rhy_samples[1:], len(rec.p_signal))
            c = Counter()
            for start, end, rhy in zip(rhy_samples, rhy_end, rhy_rhythms):
                c.update({rhy: end - start})
            total.update(c)
            print(record_name, c)
        print('Total', total)

    if args.print_classes:
        import deepdish as dd
        classes, features, testset, trainset = get_info(args.dataset_path)
        print(classes)
        features = dd.io.load(f'{args.train_output_dataset_path}/train.hdf5').keys()
        for hdf in ['trainlabel', 'testlabel']:
            datalabel = dd.io.load(f'{args.train_output_dataset_path}/{hdf}.hdf5')
            for feature in features:
                counts = [0] * len(classes)
                for one_hot, count in zip(*np.unique(datalabel[feature], return_counts=True, axis=0)):
                    counts[np.argmax(one_hot)] = count
                print(hdf, feature, counts)

    if args.check_equal:
        import deepdish as dd
        features = dd.io.load(f'{args.dataset_path}/train.hdf5').keys()
        for hdf in ['train', 'trainlabel', 'test', 'testlabel']:
            curr = dd.io.load(f'{args.train_output_dataset_path}/{hdf}.hdf5')
            orig = dd.io.load(f'{args.dataset_path}/{hdf}.hdf5')
            for feature in features:
                print(hdf, feature, np.array_equal(curr[feature], orig[feature]))

def main_dict(dict_args):
    args = parser.parse_args([])
    for key, value in dict_args.items():
        setattr(args, key, value)
    main(args)

if __name__ == '__main__':
    args_ = parser.parse_args()
    main(args_)

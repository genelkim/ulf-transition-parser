"""
Splits the data into train/dev/test split from an input json file and stores
the generated processed files into the provided split directories.
10 sentence round-robin.
"""

import json
import os
import argparse

def get_header(datum):
    sid = datum[0]
    sent = datum[1]
    return '; sid: {}\n; sentence: {}'.format(sid, sent)

def write_data(data, folder):
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, 'raw'), 'w') as out:
        out.write('\n'.join([d[1] for d in data]))
    with open(os.path.join(folder, 'ulf.preprocessed'), 'w') as out:
        out.write('\n\n'.join([get_header(d) + '\n' + d[2] for d in data]))
    with open(os.path.join(folder, 'amr'), 'w') as out:
        out.write('\n\n'.join([get_header(d) + '\n' + d[3] for d in data]))
    with open(os.path.join(folder, 'ulf.amr-format'), 'w') as out:
        out.write('\n\n'.join([get_header(d) + '\n' + d[3] for d in data]))
    with open(os.path.join(folder, 'all.json'), 'w') as out:
        out.write(json.dumps(data, indent=4))

def main(args):
    data = json.loads(open(args.input, 'r').read())
    
    # Split into 10 sentence chunks.
    # In each 10 chunks, the first goes to test, second to dev, rest to train.
    chunks = [data[i*10:(i+1)*10] for i in range((len(data) // 10) + 1) if len(data[i*10:(i+1)*10]) > 0]
    test = []
    dev = []
    train = []
    i = 0
    for chunk in chunks:
        if i % 10 == 0:
            test.extend(chunk)
        elif i % 10 == 1:
            dev.extend(chunk)
        else:
            train.extend(chunk)
        i += 1
   
    for path in [args.trainpath, args.testpath, args.devpath]:
        if not os.path.isdir(path):
            os.makedirs(path, exist_ok=True)
    write_data(train, args.trainpath)
    write_data(test, args.testpath)
    write_data(dev, args.devpath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser('split-data.py')
    parser.add_argument('--input', help='Input json file of data.')
    parser.add_argument('--trainpath', help='Path to output training data directory.')
    parser.add_argument('--testpath', help='Path to output test data directory.')
    parser.add_argument('--devpath', help='Path to output dev data directory.')
    args = parser.parse_args()
    main(args)


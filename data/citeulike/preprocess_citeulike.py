'''dtype=int
generate config.yaml, train.txt, valid.txt, test.txt, README
'''

import os
import yaml
import math
import argparse
import numpy as np
import pandas as pd



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ratios', type=str, default='8,1,1')
    parser.add_argument('--low4user', type=int, default=5, help='remove users whose iteracion number < low4user')
    parser.add_argument('--low4item', type=int, default=5, help='remove items whose iteracion number < low4item')

    args = parser.parse_args()
    print(args)
    
    rating_fn = 'users.dat'
    rating_df = {'u':[], 'i':[]}
    with open(rating_fn, 'r') as f:
        for idx, line in enumerate(f):
            items = line.strip().split()[1:]
            rating_df['u'].append(idx)
            rating_df['i'].append(items)
    rating_df = pd.DataFrame(rating_df)
    
    # convert user-item-list to user-item pairs
    pairs = []
    for idx, row in rating_df.iterrows():
        items = row.i
        for item in items:
            pairs.append((int(row.u), int(item)))
    pairs = np.array(pairs)
    rating_df = pd.DataFrame(pairs, columns=['u', 'i'])

    dsz = -1
    while dsz != len(rating_df):
        dsz = len(rating_df)

        # filter by user
        rating_df = rating_df.groupby('u').filter(lambda x: len(x['i']) >= args.low4user)
        rating_df = rating_df[['u', 'i']].reset_index().drop(columns=['index'])

        # filter by item
        rating_df = rating_df.groupby('i').filter(lambda x: len(x['u']) >= args.low4item)
        rating_df = rating_df[['u', 'i']].reset_index().drop(columns=['index'])

    # reindex users and items, both start from 0
    u = rating_df.u.unique().tolist()
    i = rating_df.i.unique().tolist()
    u.sort()
    i.sort()
    idxi = np.zeros(max(i) + 1).astype('int')
    for item in i:
        idxi[item] = i.index(item)
    idxu = np.zeros(max(u) + 1).astype('int')
    for user in u:
        idxu[user] = u.index(user)
    rating_df['u'] = rating_df['u'].apply(lambda x: idxu[x])
    rating_df['i'] = rating_df['i'].apply(lambda x: idxi[x])
    
    # #users, #items
    num_users = rating_df['u'].max() + 1
    num_items = rating_df['i'].max() + 1

    # split train, valid, test
    ratios = [eval(e.strip()) for e in args.ratios.split(',')]
    traingroups = []
    validgroups = []
    testgroups = []
    markers = np.cumsum(ratios)
    for _, group in rating_df.groupby('u'):
        if len(group) == 0:
            continue
        l = max(math.floor(markers[0] * len(group) / markers[-1]), 1)
        r = math.floor(markers[1] * len(group) / markers[-1])
        traingroups.append(group[:l])
        if l < r:
            validgroups.append(group[l:r])
        if r < len(group):
            testgroups.append(group[r:])

    trainiter = pd.concat(traingroups).reset_index(drop=True)
    validiter = pd.concat(validgroups).reset_index(drop=True)
    testiter = pd.concat(testgroups).reset_index(drop=True)

    # save
    df = pd.DataFrame(trainiter)
    df.to_csv('train.txt', sep='\t', index=False, header=False)
    num_train_inter = len(df)

    df = pd.DataFrame(validiter)
    df.to_csv('valid.txt', sep='\t', index=False, header=False)

    df = pd.DataFrame(testiter)
    df.to_csv('test.txt', sep='\t', index=False, header=False)

    from prettytable import PrettyTable
    table = PrettyTable(['#User', '#Item', '#Interactions', '#Train', '#Valid', '#Test', 'Density'])
    trainsize = len(trainiter)
    validsize = len(validiter)
    testsize = len(testiter)
    table.add_row([
        num_users, num_items, trainsize + validsize + testsize,
        trainsize, validsize, testsize,
        (trainsize + validsize + testsize) / (num_users * num_items)
    ])
    with open("README", "w") as f:
        f.write(str(table))

    print(f'# Users: {num_users}, # Items: {num_items}, # Interactions: {num_train_inter}')
    yaml.dump({'num_users': num_users, 'num_items': num_items, 'num_inters': trainsize + validsize + testsize, 'start_idx': 0}, open("config.yaml", "w"))

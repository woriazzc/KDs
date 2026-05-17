import math
import random
from tqdm import tqdm


def trans_int_feat(val):
  return int(math.ceil(math.log(val)**2+1))

def pre_parse_line(line, feat_list, int_feat_cnt=13, cate_feat_cnt=26, with_int_feat=False):
  fields_cnt = int_feat_cnt + cate_feat_cnt
  splits = line.rstrip('\n').split(',', fields_cnt + 1)

  # the index 0 is label
  start_index = 0 if with_int_feat else int_feat_cnt
  for idx in range(start_index, fields_cnt):
    val = splits[idx + 1]
    if val == '':
      continue
    elif idx < int_feat_cnt:
      val = int(val)
      if val > 2:
        val = trans_int_feat(val)
    else:
        val = int(val, 16)

    if val not in feat_list[idx]:
      feat_list[idx][val] = 1
    else:
      feat_list[idx][val] += 1

def parse_line(line, feat_list, int_feat_cnt=13, cate_feat_cnt=26, with_int_feat = False):
  fields_cnt = int_feat_cnt + cate_feat_cnt
  splits = line.rstrip('\n').split(',', fields_cnt + 1)

  label = int(splits[0])
  vals = []

  start_index = 0 if with_int_feat else int_feat_cnt
  for idx in range(start_index, fields_cnt):
    val = splits[idx + 1]
    if val == '':
      vals.append(0)
      continue
    elif idx < int_feat_cnt:
      val = int(val)
      if val > 2:
        val = trans_int_feat(val)
    else:
      val = int(val, 16)

    if val not in feat_list[idx]:
      vals.append(0)
    else:
      vals.append(feat_list[idx][val])
  return label, vals


if __name__ == "__main__":
  thres = 8
  int_feat_cnt = 13
  cate_feat_cnt = 26
  with_int_feat = True

  data_file = 'criteo.csv'
  out_file = open('criteo_all.csv', 'w')

  dataset = open(data_file, 'r').readlines()

  feat_list = []
  for i in range(int_feat_cnt + cate_feat_cnt):
    feat_list.append({})

  for line in tqdm(dataset):
    pre_parse_line(line, feat_list, int_feat_cnt, cate_feat_cnt, with_int_feat)

  for lst in tqdm(feat_list[:int_feat_cnt]):
    idx = 1
    for key, val in lst.items():
      lst[key] = idx
      idx += 1

  for i, lst in tqdm(enumerate(feat_list[int_feat_cnt:])):
    idx = 1
    tmp = {}
    for key, val in lst.items():
      if val < thres:
        # del lst[key]
        pass
      else:
        tmp[key] = idx
        idx += 1
    feat_list[i + int_feat_cnt] = tmp

  config = [len(e) for e in feat_list]
  print(config)

  for line in tqdm(dataset):
    key, vals = parse_line(line, feat_list, int_feat_cnt, cate_feat_cnt, with_int_feat)
    if vals is None:
      continue
    out_file.write('%s,%s\n' % (key, ','.join([str(s) for s in vals])))

  out_file.close()
  del dataset
  
  ## Split dataset
  name = "criteo_all.csv"
  test = open("criteo_test", "w")
  train = open("criteo_train", "w")
  val = open("criteo_val", "w")
  r = random.random()
  random.seed(2022)
  with open(name, "r") as all:
      count = 0
      dataset = all.readlines()
      random.shuffle(dataset)
      dataset_len = len(dataset)
      testLen = int(dataset_len * 0.1)
      valLen = int(dataset_len * 0.2)
      for data in tqdm(dataset):
          count += 1
          if count <= testLen:
              test.write(data)
          elif count <= valLen:
              val.write(data)
          else:
              train.write(data)
  test.close()
  val.close()
  train.close()

import random
from tqdm import tqdm
from datetime import datetime


def trans_date_feat(datetime_str):
  date_hr = datetime.strptime(datetime_str, '%y%m%d%H')
  return date_hr.weekday(), date_hr.hour

def pre_parse_line(line, feat_list, fields_cnt=23):
  # index 0 is id, index 1 is label
  splits = line.rstrip('\r\n').split(',')

  datatime_str = splits[2]
  weekday, hour = trans_date_feat(datatime_str)
  features = [weekday, hour] + [int(val) for val in splits[3:5]] +\
             [int(val, 16) for val in splits[5:14]] + [int(val) for val in splits[14:]]

  for idx in range(0, fields_cnt):
    val = features[idx]
    if val not in feat_list[idx]:
      feat_list[idx][val] = 1
    else:
      feat_list[idx][val] += 1

def parse_line(line, feat_list, fields_cnt):
  splits = line.rstrip('\r\n').split(',')

  label = int(splits[1])
  vals = []

  datatime_str = splits[2]
  weekday, hour = trans_date_feat(datatime_str)
  features = [weekday, hour] + [int(val) for val in splits[3:5]] + \
             [int(val, 16) for val in splits[5:14]] + [int(val) for val in splits[14:]]

  for idx in range(0, fields_cnt):
    val = features[idx]
    if val not in feat_list[idx]:
      vals.append(0)
    else:
      vals.append(feat_list[idx][val])
  return label, vals

"""
['id', 'click', 'hour', 'C1', 'banner_pos', 'site_id', 'site_domain',
       'site_category', 'app_id', 'app_domain', 'app_category', 'device_id',
       'device_ip', 'device_model', 'device_type', 'device_conn_type', 'C14',
       'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21']
"""
if __name__ == "__main__":
  thres = 5
  fields_cnt = 23

  data_file = 'avazu.csv'
  out_file = open('avazu_all.csv', 'w')

  dataset_ptr = open(data_file, 'r')

  dataset = dataset_ptr.readlines()

  feat_list = []
  for i in range(fields_cnt):
    feat_list.append({})

  for line in tqdm(dataset):
    pre_parse_line(line, feat_list, fields_cnt)

  for i, lst in tqdm(enumerate(feat_list)):
    idx = 1
    tmp = {}
    for key, val in lst.items():
      if val < thres:
        # del lst[key]
        pass
      else:
        tmp[key] = idx
        idx += 1
    feat_list[i] = tmp

  config = [len(e) for e in feat_list]
  print(config)

  for line in tqdm(dataset):
    key, vals = parse_line(line, feat_list, fields_cnt)
    out_file.write('%s,%s\n' % (key, ','.join([str(s) for s in vals])))

  out_file.close()
  del dataset

  ## Split dataset
  name = "avazu_all.csv"
  test = open("avazu_test", "w")
  train = open("avazu_train", "w")
  val = open("avazu_val", "w")
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

import os
import json
import pickle
import numpy as np
from tqdm import tqdm
from config import system_configs
from db.detection import DETECTION


class CityPerson(DETECTION):
	def __init__(self, db_config, split):
		super(CityPerson, self).__init__(db_config)

		data_dir = system_configs.data_dir
		result_dir = system_configs.result_dir
		cache_dir = system_configs.cache_dir

		self._split = split
		self._dataset = {
			"trainval": "train",
			"minival": "val"
		}[self._split]

		self._image_dir = os.path.join(data_dir, "leftImg8bit")

		self._image_file = os.path.join(self._image_dir, "{}")

		self._anno_dir = os.path.join(data_dir, "gtBboxCityPersons")

		self._data = "pedestrian"  # the sample function file
		self._mean = np.array([0.40789654, 0.44719302, 0.47026115], dtype=np.float32)
		self._std = np.array([0.28863828, 0.27408164, 0.27809835], dtype=np.float32)
		self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571], dtype=np.float32)
		self._eig_vec = np.array([
			[-0.58752847, -0.69563484, 0.41340352],
			[-0.5832747, 0.00994535, -0.81221408],
			[-0.56089297, 0.71832671, 0.41158938]
		], dtype=np.float32)

		self._cache_file = os.path.join(cache_dir, "cityperson_{}.pkl".format(self._split))
		self._load_data()
		self._db_inds = np.arange(len(self._image_ids))


	def _load_data(self):
		print("loading from cache file: {}".format(self._cache_file))
		if not os.path.exists(self._cache_file):
			print("No cache file found...")
			self._extract_data()
			with open(self._cache_file, "wb") as f:
				pickle.dump([self._detections, self._image_ids], f)
		else:
			with open(self._cache_file, "rb") as f:
				self._detections, self._image_ids = pickle.load(f)

	def _extract_data(self):
		self._image_ids = []
		self._detections = {}
		subsets = os.listdir(os.path.join(self._anno_dir, self._dataset))  #["frankfurt", "lindau", "munster"]
		for ss in subsets:
			anno_dir = '{}/{}'.format(self._dataset, ss)
			for anno in os.listdir(os.path.join(self._anno_dir, anno_dir)):
				anno_file = os.path.join(self._anno_dir, '{}/{}'.format(anno_dir, anno))
				img_id = os.path.join(anno_dir, anno.replace("gtBboxCityPersons.json", "leftImg8bit.png"))
				self._image_ids.append(img_id)
				bboxes = []
				with open(anno_file, 'r') as f:
					anno_info = json.load(f)
					objs = anno_info["objects"]
					for obj in objs:
						if obj['label'] == 'pedestrian':
							bbox = obj['bbox']
							bboxVis = obj['bboxVis']
							if bboxVis[2]*bboxVis[3] * 1.0 / bbox[2] * bbox[3] > 0.4:
								bbox = np.array(bbox)
								bbox[2:] += bbox[:2]
								bboxes.append(bbox.tolist())
				bboxes = np.array(bboxes, dtype=float)
				if bboxes.size == 0:
					self._detections[img_id] = np.zeros((0, 5))
				else:
					self._detections[img_id] = np.hstack((bboxes, np.ones((len(bboxes), 1))))

	def detections(self, ind):
		image_id = self._image_ids[ind]
		detections = self._detections[image_id]
		return detections.astype(float).copy()

	def _to_float(self, x):
		return float(":.2f".format(x))

if __name__=='__main__':
	import cv2
	os.chdir('../')

	cfg_file = os.path.join(system_configs.config_dir, 'CenterNet-52.json')
	with open(cfg_file, 'r') as f:
		configs = json.load(f)

	configs['system']['snapshot_name'] = 'CenterNet-52'
	system_configs.update_config(configs['system'])

	val_split = system_configs.val_split
	val_db = CityPerson(configs['db'], val_split)

	ind = 1
	img_file = val_db.image_file(ind)
	detections = val_db.detections(ind)
	img = cv2.imread(img_file)

	for d in detections:
		cv2.rectangle(img, (int(d[0]), int(d[1])), (int(d[2]), int(d[3])), color=(0, 0, 255))


	cv2.imshow('test', img)
	cv2.waitKey(0)
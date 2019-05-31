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

	def convert_to_dict(self, all_boxes):
		scores, bboxes, img_ids, clses = [], [], [], []
		for img_id in all_boxes:
			for cls_id in all_boxes[img_id]:
				dets = all_boxes[img_id][cls_id]
				img_ids.extend([img_id] * len(dets))
				clses.extend([cls_id] * len(dets))
				scores.append(dets[:, -1])
				bboxes.append(dets[:, :-1])
		scores = np.concatenate(scores, axis=0)
		bboxes = np.concatenate(bboxes, axis=0)
		detections = {"image_ids": img_ids,
					  "category_ids": clses,
					  "bboxes": bboxes,
					  "confidences": scores}
		return detections



	def evaluate(self, detections, ovthresh=0.5):
		image_ids 	= detections['image_ids']
		bboxes 		= detections['bboxes']
		confidences = detections["confidences"]
		category_ids= detections["category_ids"]  # only one class in our results

		# pre and rec
		sorted_ind 	= np.argsort(-confidences)
		bboxes		= bboxes[sorted_ind, :]
		image_ids 	= [image_ids[x] for x in sorted_ind]
		nd 			= len(sorted_ind)
		tp, fp 		= np.zeros(nd), np.zeros(nd)

		nps = 0
		R_dets = {}
		for id in image_ids:
			if id not in R_dets:
				R_dets[id] = np.zeros(len(self._detections[id]))
				nps += len(self._detections[id])

		for d in range(nd):
			R 		= self._detections[image_ids[d]]
			R_det 	= R_dets[image_ids[d]]
			bb 		= bboxes[d, :].astype(float)
			ovrmax	= -np.inf
			BBGT 	= R[:, :4].astype(float)

			if BBGT.size > 0:
				xmin = np.maximum(BBGT[:, 0], bb[0])
				xmax = np.minimum(BBGT[:, 2], bb[2])
				ymin = np.maximum(BBGT[:, 1], bb[1])
				ymax = np.minimum(BBGT[:, 3], bb[3])
				w = np.maximum(xmax - xmin + 1, 0.)
				h = np.maximum(ymax - ymin + 1, 0, )
				inters = w * h  # intersection
				unions = (bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) + \
						(BBGT[:, 2] - BBGT[:, 0] + 1.) * (BBGT[:, 3] -
														  BBGT[:, 1] + 1.) - inters

				ious = inters / unions
				ovrmax = np.max(ious)
				jmax = np.argmax(ious)
			if ovrmax > ovthresh:
				if R_det[jmax] == 0:
					tp[d] = 1
					R_det[jmax] = 1
				else:
					fp[d] = 1
			else:
				fp[d] = 1
		fp 		= np.cumsum(fp)
		tp 		= np.cumsum(tp)
		rec		= tp/float(nps)
		pre 	= tp/np.maximum(tp + fp, np.finfo(np.float64).eps)

		def voc_ap(rec, pre, use_07_metric=False):
			"""Compute VOC AP given precision and recall.
			If use_07_metric is true, uses the VOC 07 11-point method (default: False)"""
			if use_07_metric:
				ap = 0.
				for t in np.arange(0., 1.1, 0.1):
					if np.sum(rec >= t) == 0:
						p = 0
					else:
						p = np.max(pre[rec >= t])
					ap = ap + p / 11.
			else:
				# first append sentinel values at the end
				mrec = np.concatenate(([0.], rec, [1.]))
				mpre = np.concatenate(([0.], pre, [0.]))
				# compute the precision,
				for i in range(mpre.size - 1, 0, -1):
					mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
				i = np.where(mrec[1:] != mrec[:-1])[0]
				ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
			return ap
		print("The final evaluated AP: {}".format(voc_ap(rec, pre)))


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
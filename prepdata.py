import loadfile, preprocess
import os, cv2
import numpy as np
from tqdm import tqdm

if __name__ == '__main__':
	RES_DIR = 'wiki_aligned'
	data = loadfile.loadData('wiki', 'wiki.mat')
	data = preprocess.cleanData(data)
	data = data.iloc[:10]
	data = preprocess.loadImage(data)
	print('[PREPROC] Run face alignment...')
	# images = [preprocess.getAlignedFace(img) for img in tqdm(data['image'].tolist())]
	paths = data['full_path'].tolist()
	for (image, path) in tqdm(zip(data['image'].tolist(), paths), total=len(paths)):
		image = preprocess.getAlignedFace(image)
		folder = os.path.join(RES_DIR, path[0].split('/')[0])
		if not os.path.exists(folder):
			os.makedirs(folder)
		flname = os.path.join(RES_DIR, path[0])
		cv2.imwrite(flname, image)
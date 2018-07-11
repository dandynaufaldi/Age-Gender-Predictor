import loadfile, preprocess
import os, cv2
import numpy as np
from tqdm import tqdm

def main(db_name):
	RES_DIR = '{}_aligned'.format(db_name)
	data = loadfile.loadData('{}'.format(db_name), '{}.mat'.format(db_name))
	data = preprocess.cleanData(data)
	# data = data.iloc[:10]
	data = preprocess.loadImage(data)
	print('[PREPROC] Run face alignment...')
	paths = data['full_path'].tolist()
	for (image, path) in tqdm(zip(data['image'].tolist(), paths), total=len(paths)):
		image = preprocess.getAlignedFace(image)
		folder = os.path.join(RES_DIR, path[0].split('/')[0])
		if not os.path.exists(folder):
			os.makedirs(folder)
		flname = os.path.join(RES_DIR, path[0])
		if not os.path.exists(flname):
			cv2.imwrite(flname, image)

if __name__ == '__main__':
	main('wiki')
	main('imdb')
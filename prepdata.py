import loadfile, preprocess
import os, cv2
import numpy as np
from tqdm import tqdm

def main(db_name):
	RES_DIR = '{}_aligned'.format(db_name)
	data = loadfile.loadData('{}'.format(db_name), '{}.mat'.format(db_name))
	data = preprocess.cleanData(data)
	# data = data.iloc[:10]
	# data = preprocess.loadImage(data)
	print('[PREPROC] Run face alignment...')
	paths = data['full_path'].tolist()
	for (path) in tqdm(paths):
		flname = os.path.join('{}_crop'.format(db_name), path[0])
		image = cv2.imread(flname)
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
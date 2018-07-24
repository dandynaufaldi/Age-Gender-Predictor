import loadfile, preprocess
import os, cv2
import numpy as np
from tqdm import tqdm
import pandas as pd
def main(db_name):
	data = pd.read_csv('adience_u20.csv')
	RES_DIR = '{}_aligned'.format(db_name)
	print('[PREPROC] Run face alignment...')
	paths = data['full_path'].tolist()
	for (path) in tqdm(paths):
		flname = os.path.join('{}_crop'.format(db_name), path)
		image = cv2.imread(flname)
		image = preprocess.resizeImg(image)
		folder = os.path.join(RES_DIR, path.split('/')[0])
		if not os.path.exists(folder):
			os.makedirs(folder)
		flname = os.path.join(RES_DIR, path)
		if not os.path.exists(flname):
			cv2.imwrite(flname, image)

if __name__ == '__main__':
	# main('wiki')
	main('adience')
import requests
import pandas as pd
import time
from multiprocessing import Pool
from PIL import Image
from PIL.ExifTags import TAGS
from bs4 import BeautifulSoup
from datetime import datetime

BASE_URL = 'https://id.wikipedia.org'

def get_name_list(link):
    page = requests.get(link)
    if page.status_code == 200:
        soup = BeautifulSoup(page.content, 'html.parser')
        per_awal = soup.select('h2 span.mw-headline')
        link_nama = []
        for awal in per_awal[:-2]:
            try :
                nama = awal.parent.next_sibling.next_sibling.find_all('a')
                nama = [[a['href'], a.text] for a in nama if a['href'].split('/')[1].strip() == 'wiki']
                link_nama.extend(nama)
            except :
                pass
        return link_nama
    return None

def create_name_link():
    sources = [
        'https://id.wikipedia.org/wiki/Daftar_penyanyi_pria_Indonesia',
        'https://id.wikipedia.org/wiki/Daftar_penyanyi_wanita_Indonesia',
        'https://id.wikipedia.org/wiki/Daftar_aktor_Indonesia',
        'https://id.wikipedia.org/wiki/Daftar_aktris_film_Indonesia'
    ]
    data = pd.DataFrame()
    names = []
    links = []
    for source in sources :
        res = get_name_list(source)
        if res is not None:
            nama = [r[1] for r in res]
            link = [r[0] for r in res]
            names.extend(nama)
            links.extend(link)

    def add_wiki(row):
        return BASE_URL+row

    data['name'] = names
    data['link'] = links
    data['link'] = data['link'].map(add_wiki)
    data.to_csv('nama_link.csv', index=False)

def get_personal_info(link):
    page = requests.get(link)
    if page.status_code == 200:
        foto_name = None
        dob = None
        try :
            soup = BeautifulSoup(page.content, 'html.parser')
            th = soup.select('table.infobox tbody tr th')
            lahir = [t for t in th if t.text.strip() == 'Lahir'][0]
            # print('lahir', lahir)
            # print(lahir.next_sibling.next_sibling.findChildren('a'))
            tgl, tahun = lahir.next_sibling.next_sibling.findChildren('a')[:2]
            tgl, bulan = tgl.text.strip().split()
            # print('tgl bulan', tgl, bulan)
            ''' Jan, Feb, Mar, Apr, May, Jun, Jul, Aug, Sep, Oct, Nov, Dec '''
            eng_month = set(' Jan, Feb, Mar, Apr, May, Jun, Jul, Aug, Sep, Oct, Nov, Dec'.split(', '))
            decode_bulan = {
                'Mei' : 'May',
                'Agu' : 'Aug',
                'Okt' : 'Oct',
                'Des' : 'Dec'
            }
            bulan = bulan[:3]
            if bulan not in eng_month:
                bulan = decode_bulan[bulan]
            print(link)
            tahun = tahun.text.strip()
            dob = ' '.join([bulan, tgl, tahun])
            image_url = soup.select('table.infobox tbody tr td a.image')
            link_foto = BASE_URL + image_url[0]['href']
            foto_name = download_image(link_foto)
            return [foto_name, dob]
        except Exception as e:
            # print(e, 'error get info')
            return [foto_name, dob]
    return [None, None]
        
def download_image(link):
    page = requests.get(link)
    if page.status_code == 200:
        try :
            soup = BeautifulSoup(page.content, 'html.parser')
            img_url = soup.select('#file')[0].findChild('a')['href']
            img_url = 'https:' + img_url
            flname = './foto/' + img_url.split('/')[-1].strip()
            
            with open(flname, 'wb') as f:
                response = requests.get(img_url, stream=True)
                if not response.ok:
                    return None
                for block in response.iter_content(1024):
                    if not block:
                        break
                    f.write(block)
                return flname

        except Exception as e:
            # print(e, 'error download image')
            return None

def crawl():
    source = pd.read_csv('nama_link.csv')
    links = source['link'].values
    start = time.time()
    with Pool(10) as p:
        results = p.map(get_personal_info, links)
        full_path = [result[0] for result in results]
        dob = [result[1] for result in results]
        source['full_path'] = full_path
        source['dob'] = dob
        source.to_csv('crawl.csv', index=False)
        # print(result)
    finish = time.time()
    print('Time elapsed : {:.2f} sec'.format(finish - start))

def get_exif_taken(flname):
    try :
        img = Image.open(flname)
        info = img._getexif()
        taken = info[36867].split()[0]
        return taken
    except :
        return None

def get_date_taken():
    data = pd.read_csv('crawl.csv')
    paths = data['full_path'].values
    taken = [get_exif_taken(path) for path in paths]
    data['taken'] = taken
    data.to_csv('source.csv', index=False)

def age(dob, taken):
    try :
        dob = datetime.strptime(dob, '%b %d %Y')
        taken = datetime.strptime(taken, '%Y:%m:%d')
        return taken.year - dob.year
    except :
        return None

def calc_age():
    data = pd.read_csv('result.csv')
    data = data.dropna()
    dobs = data['dob'].values
    takens = data['taken'].values
    ages = [age(dob, taken) for (dob, taken) in zip(dobs, takens)]
    data['age'] = ages
    data = data.dropna()
    data.to_csv('indo.csv', index=False)

if __name__ == '__main__':
    calc_age()
    
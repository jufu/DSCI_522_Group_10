# author: Justin Fu
# date: 2020-11-19

"""Downloads a zip file and extracts all to a specified path

Usage: download_and_extract_zip.py --url=<url> --out_file=<out_file> 
 
Options:
<url>               URL to download zip file from (must be a zip file with no password)
<out_path>          Path (including filename) of where to extract the zip file contents to

"""

import zipfile
import requests
# from tqdm import tqdm
from zipfile import BadZipFile
from io import BytesIO
from docopt import docopt

opt = docopt(__doc__)


def main(url, out_path):
    """[summary]

    Parameters
    ----------
    url : string 
        URL to download zip file from (must be a zip file with no password)
    out_path : string
        Path to extract the zip file contents to

    Example
    ----------
    main(f"https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip", "../data/"
    """
    #if __name__ == "__main__":
    #main(opt["--url"], opt["--out_file"])

    #url = f"https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip"    

    try: 
        request = requests.get(url)
        zipdoc = zipfile.ZipFile(BytesIO(request.content))
        for name in zipdoc.namelist():
            print("Extracting... {0}{1}".format(out_path, name))
            zipdoc.extract(name, out_path)
        zipdoc.close() 
        print("Done extracting files from the ZipFile")

    except BadZipFile as b:
        print("Error: ", b)
    except Exception as e:
        print("Error: ", e)

    
    
if __name__ == "__main__":
    main(opt["--url"], opt["--out_file"])


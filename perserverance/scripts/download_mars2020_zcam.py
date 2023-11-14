#!/usr/bin/python3
'''
Rover Reconstruction Group
M2020 Download Scripts
Author: Ashwin Nedungadi
Maintainer: Ashwin Nedungadi
Email: ashwin.nedungadi@tu-dortmund.de
Licence: The MIT License
Status: Production
'''
import requests
from bs4 import BeautifulSoup
import wget
import sys
import os

def bar_progress(current, total, width=80):
    """ A function that shows the progress in bytes of the current download """

    progress_message = "Downloading: %d%% [%d / %d] bytes" % (current / total * 100, current, total)
    # Don't use print() as it will print in new line every time.
    sys.stdout.write("\r" + progress_message)
    sys.stdout.flush()

def isDirectory(url):
    """ A function that checks if the url is a directory
        :parameter url(str): the url to the directory where the images are
        :returns bool"""

    if(url.endswith('/')):
        return True
    else:
        return False

def download_data(url, output_directory):
    """ The main function which checks and downloads every file in the url recursively after checking if it's the
    relevant file type. A file counter is also incremented after each successful download and is printed at the end of
    the download process.
    #link['href'] shows every header in the url, useful for checking which files are there
    #(isDirectory(link['href'])) gives boolean output as to if it's a directory True or file False

    :parameter
    url(str): the url to the directory where the images are
    output_directory: the ouput_directory on local host where the files will be downloaded
    :returns
    does not return anything
    """
    page = requests.get(url).content
    soup = BeautifulSoup(page, 'html.parser')

    maybe_directories = soup.findAll('a', href=True)

    img_count = 0

    for link in maybe_directories:
        # check if files end with .IMG and .xml
        if (link['href'].endswith('.png') or link['href'].endswith('.xml')):

            # workaround to avoid downloading the same file twice
            if img_count % 2 == 0:
                img_count += 1
                pass
            else:
                # If it passes the above checks now safe to download
                print("Downloading File: " + link['href'])
                # Counter for the files in directory
                img_count += 1
                # Download everything that ends with .IMG and .xml
                wget.download(url + link['href'], output_directory, bar=bar_progress)
                print("-------->Download Successful!")


    # Do a tally of how many files got downloaded
    print("Number of Image & xml files downloaded:", int(img_count)/2)


if __name__ == '__main__':

    # Set where the files have to be downloaded here
    output_directory = '/home/jay/wise/project/perserverance/images/'
    # Enter which SOL, for a single SOL : Start_SOL = End_SOL
    Start_SOL = '00003'
    End_SOL = '00003'
    # Pay attention to which camera data of m2020 you are downloading in the url below: here it's the Mast Camera
    cam = 'zcam'
    bundle = 'mars2020_mastcamz_ops_raw'


    if Start_SOL == End_SOL:
        sol = str(Start_SOL)
        url = 'https://pds-imaging.jpl.nasa.gov/data/mars2020/mars2020_mastcamz_ops_raw/browse/sol/' + sol + '/ids/edr/zcam/'

        # Begin data download
        download_data(url, output_directory)

    else:
        start = int(Start_SOL)
        end = int(End_SOL)

        print("Data from SOLS", start, "-", end, "Will be downloaded.")

        for s in range(start, end+1):
            sol = str("{:05d}".format(s))
            url = 'https://pds-imaging.jpl.nasa.gov/data/mars2020/mars2020_mastcamz_ops_raw/browse/sol/' + sol + '/ids/edr/zcam/'

            # Make a new directory for each sol
            directory = sol
            new_path = os.path.join(output_directory, directory)
            os.mkdir(new_path)
            print("Directory '% s' created" % directory)

            # Begin data download
            download_data(url, new_path)





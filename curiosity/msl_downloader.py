''' MSL data downloader for test images
    
    Run the program from Terminal as

    ~$ python3 msl_downloader -sd 18 -ed 20
    
    will download all the data from the sols 18 and 19
    -sd is the start date of the sol
    -ed is the end data if the sol
    -h for help 
    The code can download data from sol 1 till sol 88 (will be updated for all the available sol in the future)
'''


import os
import csv
import shutil
import argparse
import requests
from bs4 import BeautifulSoup

log_file_destination = os.path.join(os.path.expanduser("~"), "download_error_file.csv")
field_names = ['File Http Link', 'Local Destination', 'Error Discription']

def log_file_updater(row_data={}):
    
    global field_names, log_file_destination
    # Default operation on the csv
    csv_filemode = 'a+'


    # Create new file if it doesn't exsist
    if not os.path.exists(log_file_destination):
        csv_filemode = "w+"
        with open(log_file_destination, csv_filemode) as file:
                writer = csv.DictWriter(file, field_names)
                writer.writeheader()
        
    
    # Update contents of the file
    if row_data != {}:

        if row_data['Error Discription'].lower() != "done":
            with open(log_file_destination, csv_filemode) as file:
                writer = csv.DictWriter(file, field_names)
                writer.writerow(row_data)
    
        # Remove successfull file downloads
        elif row_data['Error Discription'].lower() == "done":
            csv_filemode = "w+"
            with open(log_file_destination, csv_filemode) as file:
                reader = csv.DictReader(file)
                data = []
                for row in reader:
                    if row['File Http Link'] != row_data['File Http Link']:
                        data.append(row)
                writer = csv.DictWriter(file, field_names)
                writer.writeheader()
                writer.writerows(data)

def get_log_file():
    data = []
    global field_names, log_file_destination
    with open(log_file_destination) as file:
        reader = csv.DictReader(file)
        data = [row for row in reader]
    return data


def check_dir(cpath):
    if not os.path.exists(cpath):
        os.mkdir(cpath)
    return os.path.abspath(cpath)

def write_file(link, file_name, c, tot):
    with requests.get(link, stream=True) as response:
                if response.status_code == 200:
                    try:
                        with open(file_name, 'wb') as outputfile:
                            shutil.copyfileobj(response.raw, outputfile)
                            if c%2==0:
                                print(f'Links processed : {c}/{tot}')
                            return True

                    except KeyboardInterrupt:
                        print(f'Download failed for {file_name}')
                        log_file_updater(
                            {
                                "File Http Link" : link,
                                "Local Destination": file_name,
                                "Error Discription" : "User interupted"
                            }
                        )
                        print("Exiting")
                        exit(0)
                    except Exception as e:
                        print(f'Download failed for {file_name}')
                        log_file_updater(
                            {
                                "File Http Link" : link,
                                "Local Destination": file_name,
                                "Error Discription" : e
                            }
                        )
                            
                else:
                    e = "Connection not successfull"
                    print(f'Download failed for {file_name}.. {e}')
                    log_file_updater(
                            {
                                "File Http Link" : link,
                                "Local Destination": file_name,
                                "Error Discription" : str(e)
                            }
                        )
def url_check(sday, eday):
    URL = []
    
    
    #check for all the available urls in the pds
    #No of MastCam repos are 28

    for mastdata in range(1, 29):

        for day in range(sday, eday):
            if day >= 1000:
                temp_day = str(day)
            elif day >= 100:
                temp_day = "0" + str(day)
            elif day >= 10:
                temp_day = "00" + str(day)
            else:
                temp_day = "000" + str(day)
            all_url = "https://pds-imaging.jpl.nasa.gov/data/msl/MSLMST_000"+ str(mastdata) +"/DATA/RDR/SURFACE/" + temp_day + "/"
            url_response = requests.get(all_url)
            if url_response.status_code == 200:
                URL.append(all_url)
    return URL


def downloader(mode:int, sday, eday)->None:
    # Mode 1 
    #   - Download from the predefined JPL url 
    #   - day : folder to be downloaded from the JPL url
    #   - Updates logfile based on the errors

    # Mode 2
    #   - Download from the logfile 
    #   - Updates logfile based on the errors

    if mode == 1:
        URL = []
        URL = url_check(sday, eday)
        if sday >= eday:
            print("Start day must be less than end day")
            exit(0)
        
        for url in URL: 
            temp_day = str(day) if day > 10 else '0' + str(day)
            print (f'Downloading from {url}')
            current_dir = check_dir(temp_day)
            print(f"Files will be saved to {current_dir}")
            img_lbl = []
            data = requests.get(url)
            content = BeautifulSoup(data.content, features="html.parser")
            tag_a = content.find_all('a')
            print('Links loaded')

            for href in tag_a:
                url_in_a = href.get('href')
                if url_in_a.endswith('.IMG') or url_in_a.endswith('.LBL'):
                    to_add = url + url_in_a
                    if to_add in img_lbl:
                        continue
                    img_lbl.append(to_add)
            c = 1
            tot = len(img_lbl)
            for link in img_lbl:
                file_name =  os.path.join(current_dir, link.split('/')[-1])
                write_file(link, file_name, c, tot)
                c += 1
            
    elif mode == 2:
        data = get_log_file()
        tot = len(data)
        for c, row in enumerate(data,1):
            link, file_name = row["File Http Link"], row["Local Destination"]
            if write_file(link, file_name, c, tot):
                log_file_updater(
                            {
                                "File Http Link" : link,
                                "Local Destination": file_name,
                                "Error Discription" : "Done"
                            }
                        )


def main():
    log_file_updater()
    parser = argparse.ArgumentParser(description="Lowlevel script to download files from JPL site")
    parser.add_argument('-m' , '--mode', type=int, help='1 - Download from URL | 2 - Download from error log file', default=1, choices=[1,2])
    parser.add_argument('-sd' , '--start_day', type=int, help='Type the day from which the download should start', default=1, choices=range(1,88))
    parser.add_argument('-ed' , '--end_day', type=int, help='Type the day from which the download should end', default=2, choices=range(2,89))
    args = parser.parse_args()
    downloader(args.mode, args.start_day, args.end_day)



if __name__ == "__main__":
    main()

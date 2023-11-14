
from multiprocessing import Pool, freeze_support
from itertools import repeat
import time
import msl_downloader


if __name__ == '__main__':
    freeze_support()
    with Pool() as pool:
        L = pool.starmap(msl_downloader.downloader, [(1, 1, 2), (1, 2, 3), (1, 3, 4)])
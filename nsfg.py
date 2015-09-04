"""This file contains code for use with "Think Stats",
by Allen B. Downey, available from greenteapress.com

Copyright 2010 Allen B. Downey
License: GNU GPLv3 http://www.gnu.org/licenses/gpl.html
"""

from __future__ import print_function

from collections import defaultdict
import numpy as np
import sys

import thinkstats2


def ReadFemPreg(dct_file='2002FemPreg.dct',
                dat_file='2002FemPreg.dat.gz'):
    """Reads the NSFG pregnancy data.

    dct_file: string file name
    dat_file: string file name

    returns: DataFrame
    """
    dct = thinkstats2.ReadStataDct(dct_file)
    preg = dct.ReadFixedWidth(dat_file, compression='gzip')
    CleanFemPreg(preg)
    return preg


def CleanFemPreg(preg):
    """Recodes variables from the pregnancy frame.

    preg: DataFrame
    """
    # mother's age is encoded in centiyears; convert to years
    preg.agepreg /= 100.0

    # birthwgt_lb contains at least one bogus value (51 lbs)
    # replace with NaN
    preg.loc[preg.birthwgt_lb > 20, 'birthwgt_lb'] = np.nan
    
    # replace 'not ascertained', 'refused', 'don't know' with NaN
    na_vals = [97, 98, 99]
    preg.birthwgt_lb.replace(na_vals, np.nan, inplace=True)
    preg.birthwgt_oz.replace(na_vals, np.nan, inplace=True)
    preg.hpagelb.replace(na_vals, np.nan, inplace=True)

    preg.babysex.replace([7, 9], np.nan, inplace=True)
    preg.nbrnaliv.replace([9], np.nan, inplace=True)

    # birthweight is stored in two columns, lbs and oz.
    # convert to a single column in lb
    # NOTE: creating a new column requires dictionary syntax,
    # not attribute assignment (like preg.totalwgt_lb)
    preg['totalwgt_lb'] = preg.birthwgt_lb + preg.birthwgt_oz / 16.0    

    # due to a bug in ReadStataDct, the last variable gets clipped;
    # so for now set it to NaN
    preg.cmintvw = np.nan


def MakePregMap(preg):
    """Make a map from caseid to list of preg indices.

    preg: DataFrame

    returns: dict that maps from caseid to list of indices into preg df
    """
    d = defaultdict(list)
    for index, caseid in preg.caseid.iteritems():
        d[caseid].append(index)
    return d


def main(script):
    """Tests the functions in this module.

    script: string script name
    """
    preg = ReadFemPreg()
    print(preg.shape)

    assert len(preg) == 13593

    assert preg.caseid[13592] == 12571
    assert preg.pregordr.value_counts()[1] == 5033
    assert preg.nbrnaliv.value_counts()[1] == 8981
    assert preg.babysex.value_counts()[1] == 4641
    assert preg.birthwgt_lb.value_counts()[7] == 3049
    assert preg.birthwgt_oz.value_counts()[0] == 1037
    assert preg.prglngth.value_counts()[39] == 4744
    assert preg.outcome.value_counts()[1] == 9148
    assert preg.birthord.value_counts()[1] == 4413
    assert preg.agepreg.value_counts()[22.75] == 100
    assert preg.totalwgt_lb.value_counts()[7.5] == 302

    weights = preg.finalwgt.value_counts()
    key = max(weights.keys())
    assert preg.finalwgt.value_counts()[key] == 6

    print('%s: All tests passed.' % script)

if __name__ == '__main__':
    main(*sys.argv)

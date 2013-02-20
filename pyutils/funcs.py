import urllib
from collections import OrderedDict
from xml.dom.minidom import parse
import numpy as np

__all__ = ['mwdust', 'dict2array', 'transpose', 'rows2dict', 'rows2array']

IRSA_BASE_URL = \
    'http://irsa.ipac.caltech.edu/cgi-bin/DUST/nph-dust?locstr={:.5f}+{:.5f}'

def mwdust(ra, dec, source='irsa'):
    """Return Milky Way E(B-V) at given coordinates.

    Parameters
    ----------
    ra : float
    dec : float
    source : {'irsa', 'map'}, optional
        Default is 'irsa', which means to make a web query of the IRSA 
        Schlegel dust map calculator.

    Returns
    -------
    val : float
        Dust value.
    """

    if source == 'irsa':
        u = urllib.urlopen(IRSA_BASE_URL.format(ra, dec))
        if not u:
            raise ValueError('URL query returned false')
        dom = parse(u)
        u.close()

        ebvstr = dom.getElementsByTagName('meanValue')[0].childNodes[0].data
        result = float(ebvstr.strip().split()[0])

        return result


def dict2array(d):
    """Convert a dictionary of lists (of equal length) to a structured
    numpy.ndarray"""

    # first convert all lists to 1-d arrays, in order to let numpy
    # figure out the necessary size of the string arrays.
    for key in d: 
        d[key] = np.array(d[key])

    # Determine dtype of output array.
    dtypelist = []
    for key in d:
        dtypelist.append((key, d[key].dtype))
    
    # Initialize ndarray and then fill it.
    firstkey = d.keys()[0]
    col_len = len(d[firstkey])
    result = np.empty(col_len, dtype=dtypelist)
    for key in d:
        result[key] = d[key]

    return result


def transpose(it):
    """transpose a 2-d iterable (list of lists or list of tuples)."""

    if len(it) == 0:
        return []
    rowlen = len(it[0])
    if any([len(row) != rowlen for row in it]):
        raise ValueError("Iterable must have all rows equal length")
    return [[row[i] for row in it] for i in range(rowlen)]


def rows2dict(rows, colnames):
    """Convert a 2-d iterable (e.g. list of lists) where each element is a row
    to an OrderedDict where each element is a column."""

    cols = transpose(rows)
    if len(cols) != len(colnames):
        raise ValueError('length of each row must match length of colnames')
    d = OrderedDict()
    for i in range(len(cols)):
        d[colnames[i]] = cols[i]
    return d

def rows2array(rows, colnames):
    """Convert a 2-d iterable (e.g. list of lists) to a structured ndarray."""
    return dict2array(rows2dict(rows, colnames))

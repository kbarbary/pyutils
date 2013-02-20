import numpy as np
import matplotlib.pyplot as plt

__all__ = ['plotlc']

bandcolors = {'u':'#ff00ff', 'g': '#0000ff', 'r': '#00f0ff',
              'i':'#00ff00', 'z': '#f6ff00', 'y': '#ff9900'} # red #ff0000

def plotlc(data, tdata, fname, date_offset=0.):
    """Plot light curve data, save to file.

    data : structured `~numpy.ndarray` or None
        Data to be plotted as points with errorbars. 
        Must contain fields {'date', 'band', 'flux', 'fluxerr', 'zp'} 
    tdata : structured `~numpy.ndarray` or None
        Data to be plotted as lines. Must contain fields:
        {'date', 'band', 'flux', 'zp'} and optionally 'fluxerr'.
"""

    fig = plt.figure()

    bands = []
    if data is not None:
        bands.extend(np.unique(data['band']).tolist())
    if tdata is not None:
        bands.extend(np.unique(tdata['band']).tolist())
    if len(bands) = 0:
        raise ValueError('No data')
    if len(bands) > 6:
        raise ValueError('More than 6 unique bands')

    for i, band in enumerate(['u', 'g', 'r', 'i', 'z', 'y']):
        if band not in bands: continue
        plt.subplot(3, 2, i + 1)
        normflux = None

        if data is not None and band in data['band']:
            idx = data['band'] == band

            zp_factor = 10. ** (-0.4 * data['zp'][idx])
            flux = data['flux'][idx] * zp_factor
            fluxerr = data['fluxerr'][idx] * zp_factor
            
            normflux = flux.max()
            flux /= normflux
            fluxerr /= normflux

            plt.errorbar(data['date'][idx], flux, fluxerr, ls='None',
                         marker='.', markersize=3., color=bandcolors[band])

        if tdata is not None and band in tdata['band']:
            idx = tdata['band'] == band

            zp_factor = 10. ** (-0.4 * tdata['zp'][idx])
            flux = tdata['flux'][idx] * zp_factor
            
            if normflux is None: normflux = flux.max()
            flux /= normflux

            if 'date' in tdata.dtype.names:
                date = tdata['date'][idx] + date_offset
            elif 'phase' in tdata.dtype.names:
                date = tdata['phase'][idx] + date_offset
            else:
                raise ValueError("tdata must contain either 'date' or 'phase'")

            if 'fluxerr' in tdata.dtype.names:
                fluxerr = tdata['fluxerr'][idx] * zp_factor
                fluxerr /= normflux
                lower = flux - fluxerr
                upper = flux + fluxerr
                plt.fill(np.concatenate([date, date[::-1]]),
                         np.concatenate([lower, upper[::-1]]),
                         fc=bandcolors[band], alpha=0.5, ec='None')

            plt.plot(date, flux, ls='-', marker='None',
                     color=bandcolors[band])
        plt.text(0.9, 0.9, band, ha='right', va='top')
        plt.ylim(ymin=-0.1)
        
    plt.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.97,
                        wspace=0.2, hspace=0.2)
    plt.savefig(fname)
    plt.clf()


def effhist(x, success, bins=10, range=None, full_errors=False,
            return_all=False):
    """Put data into arrays that represent an 'efficiency histogram'.

    The data are split into bins and the success fraction in each bin is 
    computed, with errors.

    Parameters
    ----------
    x : list_like
        Values
    success : list_like (bool)
        Success (True) or failure (False) corresponding to items in 'x'
    bins : int or sequence of scalars, optional
        If bins is an int, it defines the number of equal-width bins
        in the given range (10, by default). If bins is a sequence, it
        defines the bin edges, including the rightmost edge, allowing
        for non-uniform bin widths.
    range : (float, float), optional
        The lower and upper range of the bins. If not provided, range
        is simply (a.min(), a.max()). Values outside the range are
        ignored.
    full_errors : bool (optional)
        Do full (correct) error calculation, returning a 68.3% confidence
        interval. Default is to
        do rough error bars (approximately correct in limit where
        efficiency is not too close to 0 or 1 and N is fairly large).
        This is much faster.
    
    Returns
    -------
    bins : numpy.ndarray
        Central value of each bin; bins are equal sized
    p : numpy.ndarray
        Efficiency in each bin 
    perr : numpy.ndarray
        2-d array of shape = (2, nbins) representing error on efficiency
        in each bin
    """

    import math
    import numpy as np
    from scipy.integrate import quad
    from scipy.stats.distributions import binom

    # Convert input values to 1-d ndarrays.
    x = np.ravel(x)
    success = np.ravel(success).astype(np.bool)

    # Put all values into a histogram.
    hist_x, bin_edges = np.histogram(x, bins=bins, range=range)
    
    # Put only values where success = True into the *same* histogram
    hist_success, bin_edges = np.histogram(x[success], bins=bin_edges)

    # Valid bins are those with more than zero entries.
    valid = hist_x > 0

    # Divide to get probability in each bin. Bins without entries are zero.
    p = np.zeros(hist_x.shape, dtype=np.float)
    p[valid] = (hist_success[valid].astype(np.float) /
                hist_x[valid].astype(np.float))

    # Initialize errors on p. 
    perr = np.zeros((2, len(p)), dtype=np.float)
    
    # Calculate error in each bin numerically.
    if full_errors:
        for i in range(len(p)):

            # If this bin doesn't have any entries, set error = 100%.
            if not valid[i]:
                perr[1:i] = 1.
                continue

            #get total prob below and above best estimate, which is p[i]
            int=quad(lambda e: binom.pmf(hist_success[i], hist_x[i], e),
                     p[i], 1) 
            totabove=int[0]

            int=quad(lambda e: binom.pmf(hist_success[i],hist_x[i],e),
                     0, p[i]) 
            totbelow=int[0]            

            #start from p[i], go down until we've encompassed 68.3% of 
            #total probability below p[i]
            pdfstep=0.00002
            sum=0.0
            currentp=p[i]-pdfstep/2.0
            while (sum<0.6826*totbelow):
                sum+=binom.pmf(hist_success[i],hist_x[i],currentp)*pdfstep
                currentp-=pdfstep
            perr[0,i]=p[i]-currentp

            #do the same thing going up:
            sum=0.0
            currentp=p[i]+pdfstep/2.0
            while (sum<0.6826*totabove):
                sum+=binom.pmf(hist_success[i],hist_x[i],currentp)*pdfstep
                currentp+=pdfstep
            perr[1,i]=currentp-p[i]

    # Otherwise, do a rough estimate of errors.
    else:
        perr[:, valid] = np.sqrt(p[valid] * (1 - p[valid]) /
                                 hist_x[valid].astype(np.float))
        perr[1, np.invert(valid)] = 1.  # empty bins have error = 100%

    # Calculate bin centers
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.

    if return_all:
        return bin_centers, p, perr
    else:
        return bin_centers[valid], p[valid], perr[:, valid]

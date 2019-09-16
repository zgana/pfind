# pfind.py

"""
Particle Finder

Find particles in an image.

Mike Richman 2017

"Tracking" / finding based on Matlab code originally by:
    David G. Grier
    Eric R. Dufresne
    John C. Crocker
    http://site.physics.georgetown.edu/matlab/code.html

"Linking" and tracking implementations both heavily reliant on scipy.ndimage


Example:

    import matplotlib.pyplot as plt
    from pfind import ParticleFinder

    pf = ParticleFinder('example.png', lsize=11, lnoise=1.5, lobject=9)
    x, y = pf.df.x, pf.df.y

    plt.imshow(pf.im, cmap='Greys_r')
    plt.scatter(x, y, color='cyan', s=1)
"""


try:
    import cv2
except:
    cv2 = False
import numpy as np
import pandas as pd
from scipy import ndimage, stats
from scipy.signal import convolve2d

from matplotlib.pyplot import imread


import progressbar


class ParticleFinder(object):

    """
    Callable for finding functions in an image.
    """

    def __init__(self, im,
                  lsize=9, lnoise=1.0, lobject=0.0, pscale=5, threshold=0,
                  pre='smoothing', post='',
                  x='x', y='y', intensity='intensity', size='size'
                 ):
        """
        Create a ParticleFinder.

            im(array or str):
                image array or filename for one
            lsize(int):
                approximate particle size in pixels(for rejecting duplicate
                intensity peaks)
            lnoise(float):
                Gaussian smoothing radius in pixels
            lobject(float):
                approximate particle size in pixels(for the high pass filter)
            pscale(float):
                percentage empty and saturated for scaling
            threshold(float):
                minimum pixel value
            pre(str):
                space-separated list of [opening, scaling, smoothing] options
                for image pre-processing(before peak finding)
            post(str):
                space-separated list of [opening, scaling, smoothing] options
                for image post-processing(after peak finding)
            x, y, intensity, size(str):
                names of DataFrame arrays for these quantities

        Example for pre or post: pre='opening scaling smoothing'
        """
        # save parameters
        self.lsize = lsize
        self.lnoise = lnoise
        self.lobject = lobject
        self.pscale = pscale
        self.threshold = threshold
        self._ims = []

        # load image if necessary
        if isinstance(im, str):
            im = 1.0 * imread(im)
        else:
            im = np.asarray(im, dtype=float)

        # store initial image
        self._ims.append(im)

        def do_processing(methods):
            for method in methods.split():
                if method == 'opening':
                    func = self.opening
                elif method == 'scaling':
                    func = self.scaling
                elif method == 'smoothing':
                    func = self.smoothing
                else:
                    raise NotImplementedError(
                        'preprocessing method "{}" '
                        'not supported'.format(method))
                self._ims.append(func())

        # perform preprocessing
        do_processing(pre)
        # find peaks, reject duplicates
        xs, ys = self.get_peaks()
        XYIR2 = self.refine_peaks(xs, ys)
        # store results as DataFrame
        columns = [x, y, intensity, size]
        self._df = pd.DataFrame(np.vstack(XYIR2).T, columns=columns)
        # perform postprocessing
        do_processing(post)


    # Data Access -----------------------------------------------------

    @property
    def ims(self):
        """All images in the processing chain."""
        return self._ims

    @property
    def im(self):
        """The image."""
        return self._ims[-1]

    @property
    def df(self):
        """The pandas.DataFrame""" 
        return self._df


    # Processing Methods ----------------------------------------------

    def opening(self):
        """Perform a grey opening: an erosion followed by a dilation."""
        def disk(N):
            """Get circle morphology."""
            y, x = np.ogrid[-N:N+.1, -N:N+.1]
            return  np.asarray(x**2 + y**2 < N**2, dtype=np.uint8)
        # obtain and subtract background lighting level
        kernel = disk(self.lsize)
        if not cv2:
             background = ndimage.grey_opening(self.im, structure=kernel)
        else:
             background = cv2.dilate(cv2.erode(self.im, kernel), kernel)
        I2 = self.im - background
        return I2

    def scaling(self):
        """Scale overall image to increase contrast."""
        im = self.im
        perc = self.pscale
        contrast_min, contrast_max = stats.scoreatpercentile(im, [perc, 100-perc])
        scale = 255. / (contrast_max - contrast_min)
        I_scaled = scale * im - contrast_min
        return np.clip(I_scaled, 0, 255)

    def smoothing(self):
        """Perform low and/or high pass filtering.

        TODO: why does this use signal.convolve2d instead of ndimage.gaussian_filter?"""
        def normed(x):
            return 1. * x / np.sum(x)
        # prep for low pass
        if self.lnoise == 0:
            gaussian_kernel = 1.0
        else:
            xmax = np.ceil(5 * self.lnoise)
            x = np.arange(-xmax, xmax + 1) / (2 * self.lnoise)
            gaussian_kernel = np.atleast_2d(normed(np.exp(-1. * x**2)))
        # prep for high pass
        if self.lobject:
            xmax = round(self.lobject)
            x = np.arange(-xmax, xmax + 1)
            boxcar_kernel = np.atleast_2d(normed(np.ones(x.shape)))
        # perform low pass
        gconv = convolve2d(self.im.T, gaussian_kernel.T, 'same')
        gconv = convolve2d(gconv.T, gaussian_kernel.T, 'same')
        # perform high pass
        if self.lobject:
            bconv = convolve2d(self.im.T, boxcar_kernel.T, 'same')
            bconv = convolve2d(bconv.T, boxcar_kernel.T, 'same')
            filtered = gconv - bconv
        else:
            filtered = gconv
        # mask out unusable border region
        lzero = int(max(np.ceil(self.lobject), np.ceil(5 * self.lnoise)))
        filtered[:,:lzero] = 0
        filtered[:,-lzero:] = 0
        filtered[-lzero:,:] = 0
        filtered[:lzero,:] = 0
        filtered[filtered < self.threshold] = 0
        return filtered


    # Peak-finding Methods --------------------------------------------

    def get_peaks(self):
        """Find local maxima in image."""
        # force size EVEN
        size = self.lsize + 1 if self.lsize % 2 else self.lsize
        sizeby2 = size // 2
        # get initial image
        im = np.copy(self.im)
        im[im <= self.threshold] = 0
        im[:sizeby2, :] = 0
        im[-sizeby2, :] = 0
        im[:, :sizeby2] = 0
        im[:, -sizeby2:] = 0
        # find initial set of peaks
        footprint = ndimage.generate_binary_structure(2, 2)
        peaks = im * (ndimage.maximum_filter(im, footprint=footprint) == im)
        i0, j0 = np.where(peaks > 0)
        # when peaks are too close together, keep only the brightest
        # NOTE: in principle, could use maximum_filter with a wider window
        # this is however MUCH slower
        for i, j in zip(i0, j0):
            left = max(0, i - sizeby2)
            right = min(im.shape[0], i + sizeby2)
            low = max(0, j - sizeby2)
            high = min(im.shape[1], j + sizeby2)
            sub_peaks = peaks[left:right,low:high]
            sub_peaks[sub_peaks != sub_peaks.max()] = 0
        y, x = np.where(peaks)
        return x, y

    def refine_peaks(self, xs, ys):
        """Refine peaks in image, weighting by nearby pixels."""
        im = self.im
        # force size ODD
        size = self.lsize if self.lsize % 2 == 1 else self.lsize + 1
        size += 2
        r = (size + 1) // 2
        # get "nearby" mask
        i, j = np.ogrid[-r+1:r, -r+1:r]
        dist = np.sqrt(i**2 + j**2)
        mask = dist < r
        dist2 = mask * dist**2

        # window x,y coords
        window_x, window_y = np.meshgrid(1. + np.r_[:size], 1. + np.r_[:size])

        # remove peaks too close to edge
        keep_peaks = (1.5 * size < xs) & (xs < im.shape[1]) \
                   & (1.5 * size < ys) & (ys < im.shape[1])
        xs, ys = xs[keep_peaks], ys[keep_peaks]

        # refine peak positions
        onorms, oxs, oys, org2s = [], [], [], []
        for (x, y) in zip(xs, ys):
            # get sub image
            left = max(0, y - r + 1)
            right = min(im.shape[0], y + r)
            low = max(0, x - r + 1)
            high = min(im.shape[1], x + r)
            sub_im = im[left:right, low:high]
            norm = np.sum(sub_im)
            # find weighted average position
            x_avg = np.sum(sub_im * window_x) / norm
            y_avg = np.sum(sub_im * window_x) / norm
            # find square of radius of gyration
            rg2 = np.sum(sub_im * dist2) / norm
            # store results
            onorms.append(norm)
            oxs.append(x + x_avg - r)
            oys.append(y + y_avg - r)
            org2s.append(rg2)

        return np.array(oxs), np.array(oys), \
               np.array(onorms), np.array(org2s)


def link(df, every=None, smooth=3, threshold=.25, output='particle', want_maps=False):
    """
    Link particle trajectories.

        df(DataFrame):
            x, y, frame, for each particle, for each frame
        every:
            process in batches of n=`every` frames
        smooth:
            number of pixels/timesteps for Gaussian smoothing
        threshold:
            arbitrary small-ish number for bins to count towards labels
        output:
            name of particle id column to add to `df`
        want_maps:
            return diagnostic maps

    Returns:
        if want_maps:
            df, (bx, by, maps)
            # plot particle mapping for first batch:
            # plt.pcolormesh(bx, by, maps[0].T)
        else:
            df  
            # just the dataframe, now with particle ids

    Notes:
        -   Smoothed histograms of particle positions for batches of frames are
        used to find allowed particle positions.  As long as `every` is
        "small" compared to the time it takes for particles to flow to other
        particles' former positions, this should let us link moving particles
        even if they are missing in individual frames
        -   TODO: binning parameters could be generalized for more arbitrary
        x,y units.
    """
    # dense histogram bins
    bx = np.arange(
        int(np.nanmin(df.x.values)) - 1,
        2 + int(np.nanmax(df.x.values)),
        .5)
    by = np.arange(
        int(np.nanmin(df.y.values)) - 1,
        2 + int(np.nanmax(df.y.values)),
        .5)
    # identify frame and batch boundaries
    idx_frames = np.r_[0, np.where(np.diff(df.frame.values))[0]]
    first_frame, last_frame = df.frame.values[[0,-1]]
    if every is None:
        every = last_frame + 1
    i_low, i_high = first_frame, first_frame + every
    # loop over batches of frames
    hs = []
    while i_low < last_frame:
        # histogram particle positions within batch
        i_high = min(i_low + every, last_frame)
        idx_low, idx_high = idx_frames[[i_low, i_high]]
        h, bins = np.histogramdd(
            (df.x.values[idx_low:idx_high], df.y.values[idx_low:idx_high]),
            bins=[bx, by]
        )
        hs.append(h)
        i_low += every
    hs = np.array(hs)
    # smooth and apply threshold
    hs = ndimage.gaussian_filter(1. * hs, smooth)
    masks = np.where(hs > threshold * every / 100., 1, 0)
    # define labels
    labels = ndimage.label(masks)[0]
    labels[labels == 0] = -1
    # assign labels to particles
    imask = np.asarray(df.frame / every, dtype=int)
    ix = np.searchsorted(bx, df.x.values) - 1
    iy = np.searchsorted(by, df.y.values) - 1
    df['particle'] = labels[imask, ix, iy]
    if want_maps:
        maps = np.ma.array(masks)
        maps.mask |= maps < 0
        return df, (bx[:-1], by[:-1], maps)
    else:
        return df



# Shape DTW python package
**shapedtw-python** is an extension to the **[dtw-python](https://github.com/DynamicTimeWarping/dtw-python)** package, implementing
the shape dtw algorithm described by L. Itii and J. Zhao in their paper, which can be downloaded from here: [shapeDTW: shape Dynamic Time Warping](https://arxiv.org/pdf/1606.01601.pdf)

In addition, to enable users to fully exploit the potential of the dtw and shape-dtw algorithms in practical applications, we have enabled the use of both versions of the multidimensional 
variation of the algorithm (dependent and independent), according to the guidelines described in the paper by B. Hu, H. Jin, W. Keogh, M. Shokoohi-Yekta and J. Wang:
[Generalizing DTW to the multi-dimensional case requires an adaptive approach](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5668684/). Github repository of the project
can be found here: [shapedtw-python](https://github.com/MikolajSzafraniecUPDS/shapedtw-python)

# Introduction to the shape dtw algorithm
In order to fully understand the shape-dtw algorithm, one must first learn the methods for calculating standard dtw.
It is worth to get familiarized with classic work of S. Chiba and H. Sakoe (available online here:
[Dynamic Programming Algorithm Optimization for Spoken Word Recognition](https://www.yumpu.com/en/document/view/29791622/dynamic-programming-algorithm-optimization-for-spoken-word-), 
but you can also read this shorter, yet comprehensive guide: [An introduction to Dynamic Time Warping](https://rtavenar.github.io/blog/dtw.html)

### Shape descriptors
In case of standard DTW we use raw time series values to determine the alignment (warping) path by which two
signals (time series) can be aligned in time. Such alignment may be susceptible to local distortion and therefore
does not fully reflect the correct relationships between signals. Zhao and Itti proposed to solve this problem by 
using so-called shape descriptors instead of single points of time series:
> Yet, matching points based solely on their coordinate
values is unreliable and prone to error, therefore, DTW may
generate perceptually nonsensible alignments, which wrongly pair
points with distinct local structures (...). This partially
explains why the nearest neighbor classifier under the DTW
distance measure is less interpretable than the shapelet classifier
[35]: although DTW does achieve a global minimal score, the
alignment process itself takes no local structural information into
account, possibly resulting in an alignment with little semantic
meaning. In this paper, we propose a novel alignment algorithm,
named shape Dynamic Time Warping (shapeDTW), which enhances DTW by incorporating point-wise local structures into the
matching process. As a result, we obtain perceptually interpretable
alignments: similarly-shaped structures are preferentially matched
based on their degree of similarity. (...)
> 
> Itti, L.; Zhao, J., shapeDTW: shape Dynamic Time Warping, 
> Pattern Recognition, Volume 74, pp. 171-184, Feb 2018.

According to Zhao and Itti shape descriptor *encodes local structural information
around the temporal point t<sub>i</sub>*. In order to calculate shape descriptor,
as a first step we need to retrieve the **subsequences** for all points of given time series.
Subsequences are simply a subsets of time series, representing neighbourhood of particular
temporal observation, which is a central point of subsequence. As a next step we calculate shape
descriptors for all the subsequences. Shape descriptor is simply a function applied to given
subsequence, which allows to properly describe its local shape properties (like slope, mean
values, wavelet coefficients, etc.). The most simple shape descriptor might be a raw subsequence
itself. Finally, we calculate the distance matrix - required by dtw algorithm - based on obtained
shape descriptors instead of raw, single values of time series. 

#### Types of shape descriptors
- Raw subsequence - raw subsequence without any transformation applied. 
- PAA - subsequence is split into *m* intervals. For each interval we calculate mean value of temporal points falling into it. Vector of such mean values is our shape descriptor.
- DWT - Discrete Wavelet Transform
- Slope - similarly as in case of PAA we split subsequence into *m* interval and fit a line according to points falling within each interval. The slopes of lines are our shape descriptor. This type of shape descriptor is invariant to y-shift.
- Derivative - first order derivative of given subsequence

All shape descriptors listed above are described in details in Zhao and Itti [paper](https://arxiv.org/pdf/1606.01601.pdf).

#### Example
Let's assume that a simple time series consisting of 4 observation is given. We would like
to acquire slope shape descriptors for it.
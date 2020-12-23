# Define utility functions for preprocessing SAR data.
import numpy as np

def _conv_timeofday(in_t):
    """Converts data acquisition time
    Args:
        in_t: time of data acquisition in format hours since 2010-01-01T00:00:00Z UTC
    Returns:
        Encoding of time where 00:00 and 24:00 are -1 and 12:00 is 1
    """
    in_t = in_t%24
    return 2*np.sin((2*np.pi*in_t)/48)-1

def _conv_deg(in_angle, is_inverse=False, in_cos=None, in_sin=None):
    """Converts measurements in degrees (e.g. angles), using encoding proposed at https://stats.stackexchange.com/a/218547
       Encode each angle as tuple theta as tuple (cos(theta), sin(theta)), for justification, see graph at bottom
    Args:
        coord: measurement of lat/ long in degrees
    Returns:
        tuple of values between -1 and 1
    """
    if is_inverse:
        return np.sign(np.rad2deg(np.arcsin(in_sin))) * np.rad2deg(np.arccos(in_cos))
    
    angle = np.deg2rad(in_angle)
    return (np.cos(angle), np.sin(angle))

def conv_real(x):
    """Scales real part of spectrum.
    Args:
        real: numpy array of shape (notebooks, 72, 60)
    Returns:
        scaled
    """
    assert len(x.shape) == 3
    assert x.shape[1:] == (72, 60)
    x = (x - 8.930369) / 41.090652 
    return x

def conv_imaginary(x):
    """Scales imaginary part of spectrum.
    Args:
        real: numpy array of shape (notebooks, 72, 60)
    Returns:
        scaled
    """
    assert len(x.shape) == 3
    assert x.shape[1:] == (72, 60)
    x = (x - 4.878463e-08) / 6.4714637
    return x
 

def median_fill(x, extremum=1e+15):
    """
    Inplace median fill.
    Args:
    x: numpy array of shape (notebooks, features)
    extremum: threshold for abs value of x. Damn Netcdf fills in nan values with 9.96921e+36.
    Returns:
    rval: new array with extreme values filled with median.
    """
    assert not np.any(np.isnan(x))
    medians = np.median(x, axis=0)
    mask = np.abs(x) > extremum
    medians = np.repeat(medians.reshape(1,-1), x.shape[0], axis=0)
    assert medians.shape == x.shape, medians.shape
    x[mask] = medians[mask] # TODO: MODIFIES x, so this is unsafe.
    return x

def conv_cwave(x):
    """
    Scale 22 cwave features. These were precomputed using following script.
    
    from sklearn import preprocessing
    with h5py.File('aggregate_ALT.h5', 'r') as fs:
        cwave = np.hstack([fs['S'][:], fs['sigma0'][:].reshape(-1,1), fs['normalizedVariance'][:].reshape(-1,1)])
        cwave = scripts.median_fill(cwave) # Needed to remove netcdf nan-filling.
        s_scaler = preprocessing.StandardScaler()
        s_scaler.fit(cwave) # Need to fit to full data.
        print(s_scaler.mean_, s_scaler.v)
    
    """
    # Fill in extreme values with medians.
    x = median_fill(x)
    
    means = np.array([ 8.83988852e+00,  9.81496891e-01,  2.04964720e+00,  1.05590932e-01,
        -6.00710228e+00,  2.54775182e+00, -5.76860655e-01,  2.09000078e+00,
        -8.44825896e-02,  8.90420253e-01, -1.44932907e+00, -6.79597846e-01,
         1.03999407e+00, -2.09475628e-01,  2.76214306e+00, -6.35718150e-03,
        -8.09685487e-01,  1.41905445e+00, -1.85369068e-01,  3.00262098e+00,
        -1.06865787e+01,  1.33246124e+00])
    
    vars = np.array([ 9.95290027, 35.2916408 ,  8.509233  , 10.62053105, 10.72524569,
         5.17027335,  7.04256618,  3.03664677,  3.72031389,  5.92399639,
         5.31929415,  8.26357553,  1.95032647,  3.13670466,  3.06597742,
         8.8505963 , 13.82242244,  1.43053089,  1.96215081, 11.71571483,
         27.14579017,  0.05681891])
    
    x = (x - means) / np.sqrt(vars)
    return x

def conv_dx(dx):
    """
    Scale dx (distance between SAR and ALT) by std. Computed with:
    
    with h5py.File('aggregate_ALT.h5', 'r') as fs:
        dd = np.hstack([fs['dx'][:].reshape(-1,1), fs['dt'][:].reshape(-1,1)])
        print(dd.std(axis=0))
    """
    return dx / 55.24285431 

def conv_dt(dt):
    """
    Scale dt (time diff between SAR and ALT) by std. Computed with:
    
    with h5py.File('aggregate_ALT.h5', 'r') as fs:
        dd = np.hstack([fs['dx'][:].reshape(-1,1), fs['dt'][:].reshape(-1,1)])
        print(dd.std(axis=0))
    """
    return dt / 36.70367443

def conv_position(latSAR):
    """
    Return cosine and sine to latitute/longitude feature.
    """
    coord_transf = np.vectorize(_conv_deg)
    cos, sin = coord_transf(latSAR)
    return np.column_stack([cos, sin])
    
def conv_time(timeSAR):
    """
    Return time of day feature.
    """
    time_transf = np.vectorize(_conv_timeofday)
    time_of_day = time_transf(timeSAR)
    #return np.column_stack(timeSAR, time_of_day)
    return time_of_day
    
def conv_incidence(incidenceAngle):
    """
    Return two features describing scaled incidence angle and 
    the wave mode label (0 or 1). Wave mode is 1 if angle is > 30 deg.
    """    
    incidenceAngle[incidenceAngle > 90] = 30
    lbl = np.array(incidenceAngle > 30, dtype='float32')
    incidenceAngle = incidenceAngle / 30.
    return np.column_stack([incidenceAngle, lbl])

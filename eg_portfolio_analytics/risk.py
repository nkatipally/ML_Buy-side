import pandas as pd
import scipy.stats
import numpy as np

# Create a function to repeat this calculation for small cap
def drawdowns(return_series:pd.Series):
    '''
    Takes a time series of asset returns computes and 
    returns a dataframe that contains:
        wealth index
        previous peaks
        percent downs
        
    '''
    wealth_index=1000*(1+return_series).cumprod()
    previous_peaks=wealth_index.cummax()
    drawdown=(wealth_index-previous_peaks)/previous_peaks
    return pd.DataFrame({
        "Wealth": wealth_index,
        "Peaks":previous_peaks,
        "Drawdown":drawdown
    })

def get_ffme_returns():
    '''
    Load the fama-French Dataset for the returns of top and bottom deciles by MarketCap
    '''
    # Read source file data
    data=pd.read_csv(r'C:\data\Portfolios_Formed_on_ME_monthly_EW.csv',
                 header=0, index_col=0, parse_dates=True, na_values=-99.99 )
    # Filter data to selected columns
    returns=data[['Lo 10', 'Hi 10']]
    #Rename columns
    returns.columns=['SmallCap10', 'LargeCap10']
    # get in percent
    returns= returns/100
    #  format returns index to datetime
    returns.index=pd.to_datetime(returns.index, format="%Y%M").to_period('M')
    return returns

def get_hfi_returns():
    '''
    Load hedge fund index returns
    '''
    hfi=pd.read_csv(r'C:\data\edhec-hedgefundindices.csv',
                    header=0, index_col=0, parse_dates=True)
    hfi=hfi/100
    hfi.index=hfi.index.to_period('M')
    return hfi

def semideviation(r):
    '''
    Returns the semideviation aka negative semideviation of r
    r must be a series or a Datframe
    '''
    is_negative=r<0
    return r[is_negative].std(ddof=0)

def skewness(r):
    '''
    Alternative to Scipy.stats.skew():
    Computes the skewness of the supplied dataframe
    Returns a float or a series
    
    '''
    demean_r=r-r.mean()
    # use the population standard deviation, so set degree of freedom dof=0
    sigma_r=r.std(ddof=0)
    exp=(demean_r**3).mean()
    return exp/sigma_r**3

def kurtosis(r):
    '''
    Alternative to Scipy.stats.kurtosis():
    Computes the kurtosis of the supplied dataframe
    Returns a float or a series
    
    '''
    demean_r=r-r.mean()
    # use the population standard deviation, so set degree of freedom dof=0
    sigma_r=r.std(ddof=0)
    exp=(demean_r**3).mean()
    return exp/sigma_r**3

def is_normal(r, level=0.01):
    '''
    Applies Jarque-Bera test to determine if a series is normal or not
    Test is applied at 1% level by default
    Returns True if the hypothesis of normality is accepted, False otherwise
    '''
    statistic, p_value=scipy.stats.jarque_bera(r)
    return p_value>level

def var_historic(r, level=5):
        '''
        Returns the historic value at risk (VaR) at a specified level
        i.e. returns the number such that "level" percent of the returns
        falls below that number and the (100-level) percent are above.
        '''
        if isinstance(r, pd.DataFrame):
            return r.aggregate(var_historic, level=level)
        elif isinstance(r, pd.Series):
            return -np.percentile(r, level)
        else:
            raise TypeError("Expected r to be Series or DataFrame")

from scipy.stats import norm
def var_gaussian(r, level=5):
    '''
    Returns the parametric Gaussian VaR of a Series or a DataFrame
    '''
    # Compute the Z score assuming it was Gaussian
    z=norm.ppf(level/100)
    return -(r.mean() +z*r.std(ddof=0))

from scipy.stats import norm
def var_cornishfisher(r, level=5, modified=False):
    '''
    Returns the parametric Gaussian VaR of a Series or a DataFrame
    If "modified" is True, then the modified VaR is returned using
    the Cornish-Fisher modification
    '''
    # Compute the Z score assuming it was Gaussian
    z=norm.ppf(level/100)
    if modified:
        # modify the Z score based on observed skewness and kurtosis
        s = skewness(r)
        k=  kurtosis(r)
        z= (z +
                (z**2-1)*s/6 +
                (z**3-3*z)*(k-3)/24 -
                (2*z**3 -5*z)*(s**2)/36
           )
    return -(r.mean() +z*r.std(ddof=0))

def cvar_historic(r, level=5):
    '''
    Computes the conditional VaR of Series or DataFrame
    '''
    if isinstance(r, pd.Series):
        is_beyond=r<=-var_historic(r, level=level)
        return -r[is_beyond].mean()
    elif isinstance(r, pd.DataFrame):
        return r.aggregate(cvar_historic, level=level)
    else:
        raise TypeError("Expected r to be a Series or DataFrame")


    
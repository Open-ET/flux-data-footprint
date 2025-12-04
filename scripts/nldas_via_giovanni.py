"""
giovanni api access to NLDAS2 data

The program is created from the tutorial found here:
https://github.com/nasa/gesdisc-tutorials/blob/main/notebooks/How_to_Access_GiC_Time_Series_Service.ipynb

Iinformation about the giovanni cloud data access is here:
https://disc.gsfc.nasa.gov/information/documents?title=Giovanni%20In%20The%20Cloud:%20Time%20Series%20Service

"""

import netrc
import requests
from requests.auth import HTTPBasicAuth
import os
import io
import pandas as pd
import numpy as np
from pathlib import Path


def get_nldas2_data(lat, lon, time_start, time_end, data, variables, out_dir, api_token):
    """
    INPUTS:
    lat - latitude
    lon - longitude
    time_start - start of time series in YYYY-MM-DDThh:mm:ss format (UTC)
    end_time - end of the time series in YYYY-MM-DDThh:mm:ss format (UTC)
    data - name of the data for requesting time series e.g. "NLDAS_FORA0125_H_2_0"
    variables - name of requesting data e.g. "Rainf", "LWdown", "SWdown", "PotEvap", "PSurf", "Qair", "Tair", "Wind_E", "Wind_N"
    out_dir - empty string will not save any file otherwise save "nldas.csv" in out_dir
    api_token - if api_token is not passed, try to get it from the three files set in user's root direcotry see _return_api_key
    
    OUTPUTS:
    df - tpandas dataframe with utc timestamp index
    """
    
    # if api_token is not passed, try to get it from the three files set in user's root direcotry
    token = _return_api_key(api_token)
    # Need to check if token is valid
    
    merged_df = pd.DataFrame()
    for var in variables:
        print(f"Requesting {var} data")
        data_var = data + "_" + var
        ts = _call_time_series(lat,lon,time_start,time_end,data_var, token)
    
        # parse data
        headers,df = _parse_csv(ts)
        if df is None:
            print(f"ERROR: No {var} data is returned!")
            continue
        
        print(f"Request for {var} data is done.")  
        merged_df = pd.merge(merged_df, df, left_index=True, right_index=True, how="outer")
    
    # save to file
    if out_dir != "": 
        if not out_dir.is_dir():
            out_dir.mkdir(parents=True, exist_ok=True)
        nldas_ts_outf = out_dir/ f'nldas.csv'
        merged_df.round(4).to_csv(nldas_ts_outf)
        print(f"NLDAS data is saved in {nldas_ts_outf}")

    return merged_df


def _return_api_key(token):
    """
    INPUT: 
    token - string if you have your api key otherwise it expects three files to be in the root directory

    Instruction for API Key:
    set giovanni api access token
    you must either get the access token by signning in here: https://api.giovanni.earthdata.nasa.gov/signin, or
    create three files, .netrc, .dodsrc and .urs_cookies in your root directory (see below for content of files) to obtain it programatically
    
    Before using giovanni api, you must approve "NASA GESDISC DATA ARCHIVE" at Applications menu in your NASA profile
    Acceess to your profile: https://urs.earthdata.nasa.gov/profile 
    
    File Contents:
    .netrc
    machine urs.earthdata.nasa.gov login < uid > password < password >
    .dodsrc
    HTTP.NETRC=< YourHomeDirectory >/.netrc
    HTTP.COOKIEJAR=< YourHomeDirectory >/.urs_cookies
    .urs_cookies is empty. This file will stores cookies related to authentication when accessing NASA Earthdata servers 
    """
    token = token
    
    # Obtain api token
    if not token: 
        signin_url = "https://api.giovanni.earthdata.nasa.gov/signin"
        token = requests.get(signin_url, auth=HTTPBasicAuth(netrc.netrc().hosts['urs.earthdata.nasa.gov'][0], 
                                                            netrc.netrc().hosts['urs.earthdata.nasa.gov'][2]),
                             allow_redirects=True).text.replace('"','')
        # print(f"API Token: {token}")
        return token

def _call_time_series(lat,lon,time_start,time_end,data,token):
    """
    INPUTS:
    lat - latitude
    lon - longitude
    time_start - start of time series in YYYY-MM-DDThh:mm:ss format (UTC)
    end_time - end of the time series in YYYY-MM-DDThh:mm:ss format (UTC)
    data - name of the data parameter for the time series
    
    OUTPUT:
    time series csv output string
    """
    time_series_url = "https://api.giovanni.earthdata.nasa.gov/timeseries"
    query_parameters = {
        "data":data,
        "location":"[{},{}]".format(lat,lon),
        "time":"{}/{}".format(time_start,time_end)
    }
    headers = {"authorizationtoken":token}
    response=requests.get(time_series_url,params=query_parameters,headers=headers)
    return response.text


def _parse_csv(ts):
    """
    INPUTS:
    ts - time series output of the time series service
    
    OUTPUTS:
    headers,df - the headers from the CSV as a dict and the values in a pandas dataframe with utc timestamp index
    """
    with io.StringIO(ts) as f:
        # the first 13 rows are header
        headers = {}
        for i in range(13):
            line = f.readline()
            # check if error is returned
            if "{" in line:
                f.readline()
                print("ERROR getting data:")
                print(f.readline())
                print(f.readline())
                return None, None

            key,value = line.split(",")
            headers[key] = value.strip()
            

        # Read the csv proper
        df = pd.read_csv(
            f,
            header=1,
            names=("date", headers["param_short_name"]),
            index_col='date',
            parse_dates=True
            # converters={"Timestamp":pd.Timestamp}
        ).sort_index()

    return headers, df
# -*- coding: utf-8 -*-
"""
Download hourly NLDAS-2 forcing data for a point from NASA Giovanni.

The code is based on the NASA GES DISC tutorial notebook:
https://github.com/nasa/gesdisc-tutorials/blob/main/notebooks/How_to_Access_GiC_Time_Series_Service.ipynb

More information about the Giovanni in the Cloud time series service:
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
    Download hourly NLDAS-2 time series for a point from NASA Giovanni.

    Each variable is requested from the Giovanni time series service and the
    variables are merged into one table. If no data comes back for a
    variable, an ERROR message is printed and that variable is skipped.
    Requires a NASA Earthdata login, see :func:`_return_api_key`.

    Arguments:
        lat (float): latitude of the point [decimal degrees].
        lon (float): longitude of the point [decimal degrees].
        time_start (str): start of the time series in UTC, as
            'YYYY-MM-DDThh:mm:ss'.
        time_end (str): end of the time series in UTC, as
            'YYYY-MM-DDThh:mm:ss'.
        data (str): name of the dataset, e.g. 'NLDAS_FORA0125_H_2_0'.
        variables (list): variable names, e.g. 'Rainf', 'LWdown', 'SWdown',
            'PotEvap', 'PSurf', 'Qair', 'Tair', 'Wind_E', 'Wind_N'.
        out_dir (:obj:`pathlib.Path`): folder to save ``nldas.csv`` in, or an
            empty string to not save a file.
        api_token (str): Giovanni API token. If empty, a token is requested
            with the Earthdata login in ``~/.netrc``.

    Returns:
        merged_df (:obj:`pandas.DataFrame`): hourly values with a UTC
        timestamp index and one column per variable.

    Example:
        >>> df = get_nldas2_data(38.1, -121.5, '2017-06-15T12:00:00',
        ...                      '2017-06-16T01:00:00', 'NLDAS_FORA0125_H_2_0',
        ...                      ['Tair', 'SWdown'], '', '')

    Note:
        This function is based on the NASA GES DISC tutorial notebook
        `How_to_Access_GiC_Time_Series_Service.ipynb
        <https://github.com/nasa/gesdisc-tutorials/blob/main/notebooks/How_to_Access_GiC_Time_Series_Service.ipynb>`_.
        More information about the service is in `Giovanni in the Cloud: Time
        Series Service
        <https://disc.gsfc.nasa.gov/information/documents?title=Giovanni%20In%20The%20Cloud:%20Time%20Series%20Service>`_.
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
    Return a Giovanni API token, requesting one if none is given.

    If ``token`` is empty, a token is requested from
    https://api.giovanni.earthdata.nasa.gov/signin using the NASA Earthdata
    login in ``~/.netrc``, which needs a line like
    ``machine urs.earthdata.nasa.gov login <uid> password <password>``.
    A token can also be copied after signing in at that address. Before using
    the Giovanni API, approve "NASA GESDISC DATA ARCHIVE" under Applications
    in your Earthdata profile (https://urs.earthdata.nasa.gov/profile).

    The templates in ``scripts/files_to_put_in_root_dir`` also include
    ``.dodsrc`` and ``.urs_cookies``, which this function does not need.

    Arguments:
        token (str): Giovanni API token, or an empty string.

    Returns:
        token (str): Giovanni API token.
    """
    token = token
    
    # Obtain api token
    if not token: 
        signin_url = "https://api.giovanni.earthdata.nasa.gov/signin"
        r = requests.get(signin_url, auth=HTTPBasicAuth(netrc.netrc().hosts['urs.earthdata.nasa.gov'][0], 
                                                        netrc.netrc().hosts['urs.earthdata.nasa.gov'][2]),
                         allow_redirects=True)
        r.raise_for_status()
        # the signin service used to return the token as a quoted string,
        # it now returns JSON, e.g. {"expires_in": 86400, "token": "..."}
        try:
            token = r.json()
        except ValueError:
            token = r.text
        if isinstance(token, dict):
            token = token['token']
        token = token.replace('"','')
        # print(f"API Token: {token}")
    return token

def _call_time_series(lat,lon,time_start,time_end,data,token):
    """
    Request one variable from the Giovanni time series service.

    Arguments:
        lat (float): latitude of the point [decimal degrees].
        lon (float): longitude of the point [decimal degrees].
        time_start (str): start of the time series in UTC, as
            'YYYY-MM-DDThh:mm:ss'.
        time_end (str): end of the time series in UTC, as
            'YYYY-MM-DDThh:mm:ss'.
        data (str): dataset and variable name, e.g.
            'NLDAS_FORA0125_H_2_0_Tair'.
        token (str): Giovanni API token.

    Returns:
        text (str): the CSV text returned by the service.
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
    Parse the CSV text returned by the Giovanni time series service.

    The first 13 lines are header lines with a key and a value. If the service
    sends an error message instead of data, the message is printed and both
    returned values are None.

    Arguments:
        ts (str): CSV text from :func:`_call_time_series`.

    Returns:
        headers (dict): the header keys and values, or None after an error.
        df (:obj:`pandas.DataFrame`): values with a UTC timestamp index and one
        column named after the variable, or None after an error.
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
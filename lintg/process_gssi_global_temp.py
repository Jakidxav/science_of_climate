"""
Authors: Jakidxav, Dennis Hartmann
Description: Program to clean GISSTemp Global Temperature Record Data, offer plotting methods.
"""
import numpy as np
import pandas as pd
import scipy.stats as stats

import matplotlib.pyplot as plt

#clean data: replace missing values and drop last incomplete row
def clean(dataframe):
	#replace missing values with NaNs
	dataframe = dataframe.replace('***', np.nan)

	#drop last row since it does not contain a full year's worth of data
	dataframe = dataframe.drop(dataframe.index[-1], axis=0)

	#convert columns from type object to type 'numeric' for calculations
	for col in dataframe:
    		dataframe[col] = pd.to_numeric(dataframe[col])

	#replace NaN values with column averages
	dn_avg = np.round(dataframe.loc[:, 'D-N'].mean(), 2)
	djf_avg = np.round(dataframe.loc[:, 'DJF'].mean(), 2)

	dataframe.loc[0, 'D-N'] = dn_avg
	dataframe.loc[0, 'DJF'] = djf_avg

	return dataframe


#This function calculates the autocorrelation function for our data set for
#a 1-month time lag. This method is based off of code written by Dennis Hartmann: auto.m.
def auto(data):
	data1 = data[:-1]
	data1 = data1 - np.mean(data1)

	data2 = data[1:]
	data2 = data2 - np.mean(data2)

	top = np.sum(np.multiply(data1, data2))
	bottom = np.sqrt(np.multiply(np.sum(data1**2), np.sum(data2**2)))
	a = top / bottom
	
	if a <= 0:
		a = 0

	return a


#This method takes in a dataframe and a column name for the GSSI Global Temperature data.
#It calculates the covariance, residuals, and autocorrelation for those residuals, and returns
#the length of the input data, the variance, the parameters a0 and a1, the residual, and its autocorrelation for a 1 time-step lag.
#This method is based off of part of code written by Dennis Hartmann: lintg.m
def calculate_res_autocorr(dataframe, column_name):
	temp = dataframe[column_name]
	temp_mean = np.mean(temp)
	temp_anom = temp - temp_mean
	time = dataframe['Year'].values
	td = time - np.mean(time)
	n = len(time)

	#covariance
	ttp = np.matmul(np.matrix(td), np.matrix(temp_anom).T) / n
	vart = np.matmul(td.T, td) / n
	a1 = ttp / vart
	a0 = temp_mean - (a1 * np.mean(time))

	#compute residuals
	temp_res = temp.values - (a0 + (a1 * time))

	#convert from matrix back to numpy array
	temp_res = np.array(temp_res)[0]
	residual = np.mean(temp_res)

	#compute autocorrelation of residuals
	autocorr = auto(temp_res)

	return n, vart, a0, a1, temp_res, residual, autocorr


#This method computs the sensitivity of a given trend. To do so, it calculates the degrees
#of freedom for a given autocorrelation, and confidence limits using residuals. Then a t-test
#is performed for a given level of significance. Currently, only 95% significance is implemented.
#This method is based off of part of code written by Dennis Hartmann: lintg.m
def calculate_significance(n, a0, a1, vart, autocorr, temp_res):
	# compute confidence limits on slope
	#first compute estimated degrees of freedom from red noise model of residuals.
	dof = n * (1 - autocorr) / ( 1 + autocorr)

	#Compute the error variance adjusted for DOF
	se2 = np.matmul(temp_res.T, temp_res) / (dof - 2)
	sigb = se2 / (vart * n)

	#compute % limits on b
	critical_t = stats.t.ppf(q=0.975, df=dof - 2)
	sb2 = se2 / (n * vart)
	sb = np.sqrt(sb2)

	low_t = a1 - sb * critical_t
	high_t = a1 + sb * critical_t
	delta_a = sb * critical_t * 10
	a1_t = a1 * 10

	return dof, sb, critical_t, low_t, high_t, delta_a, a1_t



#matplotlib plotting method for plotting temperature anomaly data along with linear fit
#provided by a start and end year
def plot(start, end, dataframe, column, c, sig_level):
    #check for correct year input
    minyr = dataframe.Year.min()
    maxyr = dataframe.Year.max()
    
    start = int(start)
    end = int(end)
    
    #make sure input is in between correct years
    if np.logical_or(start<minyr, end>maxyr):
        print('Please enter integer years between 1880 and 2018!')
        return
    

    #create indexer array
    tidx = np.logical_and(dataframe.Year >= start, dataframe.Year <= end)
    
    #subset for different times
    trend = dataframe[tidx]
    
    #calculate temperature anomaly here
    anomaly = dataframe[column] - dataframe[column].mean()

    #linear regressions
    x = trend.Year
    y = trend[column]
    m, b = np.polyfit(x, y, 1)

    #calculate residuals and autocorrelation
    n, vart, a0, a1, temp_res, residual, autocorr = calculate_res_autocorr(trend, column)

    #significance testing
    dof, sb, critical_t, low_t, high_t, delta_a, a1_t = calculate_significance(n, a0, a1, vart, autocorr, temp_res)
    a1_t = float(a1_t)

    plt.figure(figsize=(12,8))
    label = '{}-{} trends, {} significance, {:.3f} $\pm$ {:.3f}, DOF = {:.2f}'.format(start, end, sig_level, a1_t, delta_a, dof)

    plt.plot(dataframe.Year, anomaly, color='k', linewidth=2, label='Anomaly Data, {}'.format(column))
    plt.plot(x, m*x+b, color=c, linewidth=2, label=label) 

        #label axes
    plt.xlabel('Year', fontsize=14)
    plt.ylabel('T Anomaly (C)', fontsize=14)

    #add legend
    plt.legend(fontsize='large')
    plt.show()

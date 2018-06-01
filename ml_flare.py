#!/usr/bin/env python
#CHANGELOG
#v0.6: all originally intended features added
#v0.7: automatic polynomial order selection using n light curves and cross-validation
#v0.71: option for magnitudes and detection sigma added
#v0.72: evaluation windows start before light curve, 
#       so early events can get enough votes
#v0.80: integrated flare energy calculated; 
#v0.81: LC t_max also saved 
#v0.82: FWHM is now a possible command line argument for fitting
#v0.83: Fixed a bug that killed GridSearchCV for polynomial degrees in data gaps
#v0.84: Bugfix with multiple input files; added debug-noplot option 
#v0.85: l286: extra point is added to flares only if it's not too far away & suppress RuntimeWarnings
#v0.86: Bugfix: false positive events with points over data gaps
#v0.87: added a few extra filters for strange events

import warnings
warnings.filterwarnings("ignore")

import numpy as np
from matplotlib import pyplot as plt
from gatspy.periodic import LombScargleFast


from sklearn import linear_model
from sklearn import preprocessing
from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold, GridSearchCV

from aflare import aflare1
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy.integrate import simps


class PolynomialRANSAC(BaseEstimator):
	'''
	scikit-learn estimator that enables tuning of polynomial degrees 
	for linear regression
	'''

	def __init__(self, deg=None):
		self.deg = deg

	def fit(self, X, y, deg=None):
		#Adding random_state=0 for consistency in consecutive runs
		self.model = linear_model.RANSACRegressor(random_state=0)
		self.model.fit(np.vander(X, N=self.deg + 1), y)

	def predict(self, x):
		return self.model.predict(np.vander(x, N=self.deg + 1))

	@property
	def coef_(self):
		return self.model.coef_
	@property
	def inlier_mask_(self):
		return self.model.inlier_mask_

def SelectDegree(time,flux,window,seed=42, n_iterate=5, debug=False):
	if debug:
		print "Selecting best degree with",n_iterate,"samples..."
	np.random.seed(seed)
	best_degree = []
#	for i in range(n_iterate):
	i = 0
	watchdog = 0
	while i < n_iterate:
		t0 = np.random.random() * ( time.max()-time.min() ) + time.min()
		ind = np.where( (time > t0) * (time < t0+window) )
		t = time[ind]
		f = flux[ind]

		t_mean = np.mean(t)
		t_std = np.std(t)
		t_scale = (t - t_mean) / t_std

		grid = GridSearchCV(PolynomialRANSAC(),
							param_grid={'deg':np.arange(1,15)}, 
							scoring='neg_median_absolute_error', 
							cv=KFold(n_splits=3,shuffle=True),
							verbose=0, n_jobs=1)

		if watchdog > 50:
			print "SelectDegree is running in circles..."
			best_degree=8
			break

		try:
			grid.fit(t_scale, f)
		except ValueError:
			watchdog +=1
			if debug:
				#This can get really messy with long gaps...
				pass
				#print "I really shouldn't be here. Is this a data gap?",watchdog
			continue


		if debug:
			print "{:2d}: {:4d} LC points, best polynomial degree: {:2d}".format( i+1, np.size(ind), grid.best_params_['deg'])
		best_degree.append( grid.best_params_['deg'] )
		i+=1

	degree = int(np.median(best_degree))
	if debug:
		print "Using polynomials of degree {:2d}".format( degree )
	return degree


def FindPeriod(time,flux, minper=0.1,maxper=30,debug=False):
	'''
	Finds period in time series data using Lomb-Scargle method
	'''
	pgram = LombScargleFast(fit_period=True,\
							optimizer_kwds={"quiet": not debug})
	pgram.optimizer.period_range = (minper, maxper)
	pgram.fit(time,flux)
	if debug:
		print "Best period:", pgram.best_period
	return pgram.best_period


def FindFlares(time,flux, period, window_var=1.5,
		       shift_var=4., degree=0, detection_sigma=3.,
			   detection_votes=3, returnbinary=True, N3=2, 
			   debug=False):
	'''
	Finds flare candidates using machine learning algorithm. 
	Currently the random sample consensus (RANSAC) method is
	implemented, as this yields a fast and robust fit. 
	The algorithm takes a given window (1.5*P_rot by default) 
	and fits a polynomial of given degree. Using RANSAC estimate 
	of the inlier points the standard deviation is calculated, 
	and the flare candites are selected.
	Since the polynomial fit might overfit the light curve at the
	ends (or RANSAC select these as outliers), this selection is 
	done multiple times by shifting the window, and only those flare 
	candidate points are kept, which get at least a given number of 
	vote.

	Parameters
	----------
	time: numpy array
		Light curve time array
	flux: numpy array
		Light curve flux array
	window_var: float, optional
		Determines the size of fit window in period units (default: 1.5)
	shift_var: float, optional
		Determines window shift, portion of window (default: 3.)
	degree: int, optional
		Size of the Vandermonde matrix for the fit, determines 
		the order of fitted polynomial (default: 10)
	detection_sigma: float, optional
		Detection level of flare candidates in 
		np.stdev(flux - model) units (default: 3.)
	detection_votes: int, optional
		Number of votes to accept a flare candidate. If shift_var is
		changed, consider changing this, too. (default: 2)
	returnbinary: bool, optional
		If True the return value is a boolean mask with flare points
		marked. Otherwise flare start/end points are returned.
	N3:	int,optional
		Number of consecutive candidate points needed for a flare event
	
	Returns
	-------
	returnbinary == True:
		boolean array with flare points flagged
	returnbinary == False:
		two arrays with flare start/end indices
	'''

	if debug:
		print "Using period:", period

	#We define the window to be fitted. Too short windows
	#might be overruled by long flares, too long ones might have
	#poor fits. 
	window = window_var * period
	shift = window / shift_var

	isflare = np.zeros_like( time )
	#We put the first windows before the light curve so not to miss
	#the early events
	t0 = np.min(time) - window + shift
	i = 0

	#You probably don't want to call this again, but who knows...
	if degree == 0:
		degree = SelectDegree(time,flux,period*1.5,debug=debug)

	#Originally the built-in RANSAC was used, but grid search for polynomial
	#degree is not possible that way...
	#regressor = linear_model.RANSAC(random_state=0)
	regressor = PolynomialRANSAC(deg=degree)

	while t0 < np.max(time):
		#degree = 10
		ind = np.where( (time > t0) * (time < t0+window) )
		#If we find a gap, move on
		if np.size(ind) <= degree+2:
			t0 += shift
			continue

		t = time[ind]
		f = flux[ind]

		#Machine learning estimators might behave badly 
		#if the data is not normalized
		t_mean = np.mean(t)
		t_std = np.std(t)
		t_scale = (t - t_mean) / t_std
		
		#Polynomial fit is achieved by feeding a Vandermonde matrix
		#to the regressor
		#regressor.fit( np.vander(t_scale, degree), f )

		#With the custom polynomial regressor we can just input time/flux
		regressor.fit( t_scale, f )
		

		#RANSAC outlier estimation is not trustworthy
		#flare_mask = np.logical_not(regressor.inlier_mask_)

		#model = regressor.predict( np.vander((t-t_mean)/t_std, degree) )
		model = regressor.predict( t_scale )

		#We won't use the outlier for statistics:
		stdev = np.std( (f - model)[regressor.inlier_mask_] )
		#Then select flare candidate points over given
		#sigma limit for this segment 
		flare_mask =  f > model + detection_sigma*stdev 

		#Each candidate gets a vote for the given time segment
		isflare[ind[0][0] : ind[0][-1]+1] += flare_mask 

#DEBUG Use this to plot the fitted model
#	t_plot = np.linspace(np.min(time[ind]),np.max(time[ind]), 1000)
#	f_plot = regressor.predict( np.vander(t_plot-t_mean,degree) )
#	f_plot = regressor.predict( np.vander((t_plot-t_mean)/t_std, degree) )

#NOTE: this is too much even for debug mode :)
#		if debug:	
#			print "Segment ",i,"\tCandidate points: ",\
#				  np.size(np.where(flare_mask==True))

		#Move on to the next segment
		t0 += shift
		i+=1

#DEBUG Use this to stop the algorithm somewhere for inspection
#		if i>=111:
#			break
#		if np.size(ind) == 0:
#			break

#	return isflare


	##############################################
	# Thankfully taking this part from appaloosa #
	# https://github.com/jradavenport/appaloosa  #
	##############################################
	#
	#We'll keep only candidates with enough votes
	ctmp = np.where( isflare >= detection_votes )

	cindx = np.zeros_like(flux)
	cindx[ctmp] = 1

	# Need to find cumulative number of points that pass "ctmp"
	# Count in reverse!
	ConM = np.zeros_like(flux)
	# this requires a full pass thru the data -> bottleneck
	for k in xrange(2, len(flux)):
		ConM[-k] = cindx[-k] * (ConM[-(k-1)] + cindx[-k])

	# these only defined between dl[i] and dr[i]
	# find flare start where values in ConM switch from 0 to >=N3
	istart_i = np.where((ConM[1:] >= N3) &
						(ConM[0:-1] - ConM[1:] < 0))[0] + 1

	# use the value of ConM to determine how many points away stop is
	istop_i = istart_i + (ConM[istart_i] - 1)


	#we add an extra point for better energy estimation...
	#... but only if it's not too far away
	for j in range( len(istart_i) ):
		try:
			if np.abs( time[ istart_i[j] ] - time[ istart_i[j-1] ] ) < \
				2*np.abs( time[ istart_i[j+1] ] - time[ istart_i[j] ] ):
				istart_i[j] -= 1
		except IndexError:
			pass


	istart_i = np.array(istart_i, dtype='int') 
	istop_i = np.array(istop_i, dtype='int')

	if returnbinary is False:
		return istart_i, istop_i
	else:
		bin_out = np.zeros_like(flux, dtype='int')
		for k in xrange(len(istart_i)):
			bin_out[istart_i[k]:istop_i[k]+1] = 1
		return np.array(bin_out, bool)

	#############
	# </thanks> #
	#############


def FitFlare(time,flux,istart,istop,period,window_var=1.5, degree=10, debug=False, domodel=True):
	
	midflare = (time[istart] + time[istop]) / 2.
	window_mask = (time > midflare - period*window_var/2.) \
				* (time < midflare + period*window_var/2.)
	t = time[window_mask]
	f = flux[window_mask]

	#We should never get here, but if we have two points 
	#across a gap for some reason...
	if np.size(t) == 0:
		t = np.linspace(0,1,5)
		f = np.ones(5)
		model = np.ones_like(f)
		fx, fy, fy0 = t, model, f-model
		degree = 1
		if domodel:
			popt1 = np.array([np.nan, np.nan, np.nan])
			stdev=np.nan
			return fx, fy, fy0, popt1, stdev, [0, 5]
		else:
			return fx, fy, fy0, 0, 0, [0, 5]


	start = (np.abs( t-time[istart] )).argmin()
	stop = (np.abs( t-time[istop] )).argmin()
	t_mean = np.mean(t)
	t_std = np.std(t)
	t_scale = (t - t_mean) / t_std
	
	#You probably don't want to call this again, but who knows...
	if degree == 0:
		SelectDegree(time,flux,period*1.5,debug=True,n_iterate=5)

	regressor = PolynomialRANSAC(deg=degree)

	regressor.fit(t_scale, f)
	model = regressor.predict(t_scale)


	#Time, flare-free LC and unspotted LC with flare
	fx, fy, fy0 = t, model, f-model
	
	if domodel:
		#We also save the stdev for calculating start/end times
		stdev = np.std( (f - model)[regressor.inlier_mask_] )
		
		try:
			global fwhm 		#I know, there is a special place in hell for this...
			if fwhm == 0: 		#not defined as command-line argument
				fwhm = 1./24 	#Selected by educated random guessing
		except NameError:
			fwhm = 1./24		#If calling just this function this might be handy

		#First try: the input is used as peak
		#tpeak = time[ind]
		#Second try: the maximum of the selection is considered as peak. 
		#But what if there are other events around?
		#ftpeak = fx[ np.argmin( np.max(fy0) - fy0) ]
		#tpeak = time[ np.argmin( np.abs( time - ftpeak ) ) ]
		tpeak = np.average( time[istart:istop+1] )

		#Same goes for the amplitude:
		#ampl = np.max(fy0)
		ampl = np.max( flux[istart:istop+1] )

		pguess = (tpeak, fwhm, ampl)
		
		try:
			popt1, pcov = curve_fit(aflare1, fx, fy0, p0=pguess)
		except ValueError:
			# tried to fit bad data, so just fill in with NaN's
			# shouldn't happen often
			popt1 = np.array([np.nan, np.nan, np.nan])
		except RuntimeError:
			# could not converge on a fit with aflare
			# fill with bad flag values
			popt1 = np.array([-99., -99., -99.])
		if debug:
			print "Initial guess:",pguess
			print "Fitted result:",popt1
		return fx, fy, fy0, popt1, stdev, [start, stop]
	else:
		return fx, fy, fy0, 0, 0, [start, stop]


def GenerateOutput(time,flux,istart,istop,period, degree=10,\
				   fit_events=True,debug=False, outputfile=""):
	if debug:
		figure = plt.figure(num=1)
		figure.clf()
		ax = plt.subplot()
		ax.scatter(time, flux, c="C0")

	if outputfile:
		fout = open(outputfile, "w")
		
	if fit_events:
		header = "{:14}{:14}{:14}{:14}{:14}{:14}{:14}{:14}{:14}{:14}{:14}".format(\
				 "#t_start",\
				 "t_end",\
				 "t_max",\
				 "flux_max",\
				 "raw_integral",\
				 "fit_amp",\
				 "fit_fwhm",\
				 "fit_t_start",\
				 "fit_t_end",\
				 "fit_t_max",\
				 "fit_integral")
	else:
		header = "{:14}{:14}{:14}{:14}{:14}".format("#t_start","t_end","t_max","lc_amp","raw_integral")
	
	if outputfile:
		fout.write(header+"\n")
	else:
		print header

	for i in xrange(len(istart)):
		t_start = time[ istart[i] ]
		t_stop = time[ istop[i]+1 ]
		#index of maximum LC point from the selection
		maxind = istart[i] + np.argmax(flux[ istart[i]:istop[i]+1 ])
		
		#For determining the flare energy, we have to integrate the light curve
		fx, fy, fy0, popt1, stdev, ind = \
			FitFlare(time,flux,istart[i],istop[i],period, degree=degree,domodel=fit_events)

		raw_integral = simps( fy0[ind[0]:ind[1]], fx[ ind[0]:ind[1] ] )

		t_max = time[ maxind ]  #We also save the time of maximum

		#For precise start/end time and amplitude we fit the event
		#NOTE: funny fits yield funny results
		if fit_events:
			flare_t = np.linspace(np.min(fx), np.max(fx), (fx[-1]-fx[0])*10000)
			flare_f = aflare1(flare_t, popt1[0], popt1[1], popt1[2] )

			amp = np.max(flare_f)
			fit_t_max = flare_t[ np.argmin( np.abs(amp - flare_f) ) ]

			fx_maxind = np.argmin( np.abs( fx - time[maxind] ) )
			lc_amp = flux[maxind] - fy[fx_maxind]

			fit_int = simps( flare_f, flare_t )

			#Flare event is defined where it is above noise level
			event_ind = np.where(flare_f > stdev )

			#Filters for strange detections:
			if ( fx[ind[-1]] - fx[ind[0]] ) / ( ind[-1]-ind[0] ) > 10*(fx[-1] - fx[0] ) / len(fx):
				#NOTE: long gap during the event -> pass
				continue
			if lc_amp < 0 :
				#NOTE: negative amplitude (WHY?) -> pass
				continue
		
			if np.size(event_ind) > 0: 
				fit_t_start = flare_t[ event_ind[0][0] ]
				fit_t_stop = flare_t[ event_ind[0][-1] ]

				outstring="{:<14.4f}".format(t_start)+\
						  "{:<14.4f}".format(t_stop)+\
						  "{:<14.4f}".format(t_max)+\
						  "{:<14.4f}".format(lc_amp)+\
						  "{:<14.8f}".format(raw_integral)+\
						  "{:<14.4f}".format(amp)+\
						  "{:<14.4f}".format(popt1[1])+\
						  "{:<14.4f}".format(fit_t_start)+\
						  "{:<14.4f}".format(fit_t_stop)+\
						  "{:<14.4f}".format(fit_t_max)+\
						  "{:<14.8f}".format(fit_int)
			else:
				#Honestly, we should never get here...
				outstring="{:<14.4f}".format(t_start)+\
						  "{:<14.4f}".format(t_stop)+\
						  "{:<14.4f}".format(t_max)+\
						  "{:<14.4f}".format(lc_amp)+\
						  "{:<14.8f}".format(raw_integral)+\
						  "{:<14.4f}".format(-99)+\
						  "{:<14.4f}".format(-99)+\
						  "{:<14.4f}".format(-99)+\
						  "{:<14.4f}".format(-99)+\
						  "{:<14.4f}".format(-99)+\
						  "{:<14.8f}".format(-99)
				pass
			
			if debug:
				#We plot the detected flare points 
				ax.scatter(time[ istart[i]:istop[i]+1 ],\
						   flux[ istart[i]:istop[i]+1 ], c="C3") 

				#The flare plot with the observed sampling would
				#be misleading, we rather interpolate for the flare-free 
				#fit region and use that 
				interp = interp1d(fx, fy)
				fi = interp(flare_t)
				ax.plot(flare_t[event_ind], (fi+flare_f)[event_ind], c="C2")

		else:
			#without the fit we can give only a crude estimate on the 
			#amplitude based on the neighbors
			lc_amp = flux[maxind]-np.median(flux[maxind-10:maxind+10])

			outstring="{:<14.4f}".format(t_start)+\
					  "{:<14.4f}".format(t_stop)+\
					  "{:<14.4f}".format(t_max)+\
					  "{:<14.4f}".format(lc_amp)+\
					  "{:<14.8f}".format(raw_integral)
			if debug:
				ax.scatter(time[ istart[i]:istop[i]+1 ],\
					   flux[ istart[i]:istop[i]+1 ], c="C3") 

		if outputfile:
			fout.write(outstring+"\n")
		else:
			print outstring

	if outputfile:	
		fout.close()

	if debug and not noplot:
		plt.show()
		plt.ion()
		_ = raw_input("<Press enter to continue>")

def usage():
	print "Usage:", sys.argv[0],"[options] <input file(s)>"
	print "Options:"
	print "-h, --help:\tprint this help message"
	print "-o, --outfile=\toutput file name (for single input file)"
	print "-n, --flarepoints=<n>\t set number of candidate points needed for a flare (default: 2)"
	print "-m, --magnitude\tlight curve is in magnitudes instead of flux units"
	print "-s, --sigma=<s>\tdetection level of flares (only if analytic fit is done)"
	print "-p, --period=<p>\tuse the given period with a search window of 1.5*p, and skip the period search"
	print "-f, --fwhm=<f>\tuse the given FWHM for flare fitting"
	print "--degree=<d>\tuse polynomials of given degree to fit light curves"
	print "--hardcopy\tsave output to <input file>.flare files"
	print "--nofit\t\tdo not fit analytic flare model to the data. Flare amplitude will be just a crude estimation based on nearby points"
	print "--debug\t\tverbose output"
	print "--debug-noplot\t\tverbose output without plot (e.g. for several input files)"
	sys.exit()

if __name__ == "__main__":
	import getopt
	import sys

	hardcopy = False
	outfile = ""
	debug = False
	noplot = False
	fit_events = True
	magnitude = False
	flarepoints = 2
	sigma = 3
	period=0.
	degree=0
	fwhm = 0.
	try:
		opts, args = getopt.getopt( sys.argv[1:], "hmo:n:s:p:f:",\
			["help", "magnitudes", "outfile=", "flarepoints=","sigma=", "period=","fwhm=","degree=","debug","debug-noplot", "hardcopy", "nofit", "no-fit"])

	except getopt.GetoptError as err:
		print str(err)
		usage()
		sys.exit()

	for opt, arg in opts:
		if opt in ("-h", "--help"):
			usage()
			sys.exit()
		elif opt == "--debug":
			debug = True
		elif opt == "--debug-noplot":
			debug = True
			noplot = True
		elif opt == "--hardcopy":
			hardcopy = True
		elif opt in ("-n", "--flarepoints"):
			flarepoints = float(arg)
		elif opt in ("-m", "--magnitudes"):
			magnitude = True
		elif opt in ("--nofit", "--no-fit"):
			fit_events = False
		elif opt in ("-s", "--sigma"):
			sigma = float(arg)
		elif opt in ("-p", "--period"):
			period = float(arg)
		elif opt in ("-f", "--fwhm"):
			fwhm = float(arg)
		elif opt == "--degree":
			degree = int(arg)
		elif opt in ("-o", "--outfile"):
			if len(args) > 1:
				print "Multiple input files with one output file, stopping"
				print "Just drop the -o option and run again"
				sys.exit()
			else:
				outfile = arg
				if debug:
					print "Outfile:", outfile

	for filename in args:
		infile = np.genfromtxt(filename)
		if debug:
			print "\n\nInput:",filename
		time = infile[:,0]
		flux = infile[:,1]
		if magnitude:
			flux = 10**(-0.4*flux)
	

		if period == 0:
			period = FindPeriod(time, flux,debug=debug)
		window_var = 1.5

		#Optionally you can use the code interactively to get a flare mask:
		#isflare = FindFlares(time,flux, period)
		#If time is not a concern, the search can be run multiple times
		#to remove false positives:
		#isflare = FindFlares(time,flux, period) *\
		#		   FindFlares(time,flux, period)


		if degree == 0:
			degree = SelectDegree(time,flux,period*window_var,debug=debug)
		istart, istop = FindFlares(time,flux, period,
								   returnbinary=False,
								   N3=flarepoints,
								   degree=degree, 
								   detection_sigma=sigma,
								   debug=debug)

		if hardcopy or outfile != "":
			if outfile == "":
				outfile = filename+".flare"
			if debug:
				print "Saving output into:", outfile
			
		GenerateOutput(time,flux,istart,istop,period,\
					   fit_events=fit_events,degree=degree,debug=debug,outputfile=outfile)

		#In case there are more input files, reinitalize:
		outfile = ""
		period=0.
		degree=0
		fwhm = 0.

		for opt, arg in opts:
			if opt in ("-p", "--period"):
				period = float(arg)
			elif opt in ("-f", "--fwhm"):
				fwhm = float(arg)
			elif opt in ("--degree"):
				degree = int(arg)


#plt.clf()
#plt.scatter(time, flux)
#plt.scatter(time[isflare],flux[isflare])

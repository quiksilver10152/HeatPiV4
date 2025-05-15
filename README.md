# HeatPiV4
An analysis app that takes an input folder of EIS files containing frequency, real, and imaginary impedance values then runs 1 of 4 different methods to generate heatmaps and .csv files.

HeatPi EIS Data Processing Read Me

Folder Setup
	Create a project folder somewhere on your computer with whatever title you wish.
	Inside this folder create a separate folder for each of your replicates, naming them anything.
	Each replicate folder should contain a sequence of measurements with specific naming. The first file (bare or post-modification) should be labeled ‘0.DTA’ with each subsequent measurement being numbered starting at 1.
	The filepath for the project folder should be entered as the input (and possibly output) on HeatPi.

Library Installation
deareis==4.2.1
matplotlib==3.8.4
numpy==1.24.3
pandas==2.0.3
seaborn==0.13.2

App Details
	Global Settings:
 
	Input Directory
	The folder created at the top of this document.
 
	Output Directory
	The folder where the generated heatmaps will be created.
 
	Replicates
	Only used if there are additional files in the project folder. Try not to do that.
 
	Apply day0 Normalization
	Should the first file of each replicate be subtracted from all subsequent values after analysis?
	Useful for subtracting against bare, first titration, pre-stimulus, or post modification changes.
 
	Average Replicates
	Should analysis methods run on each replicate separately or should they be averaged first?
	Generates individual heatmaps and csv files if this option isn’t checked.
 
	Heatmap Norm Strategy
	Raw will display full range of values and tends to drown out most of the structure. Useful for analyzing relative changes using day0.
	By Parameter will normalize across the timeline of each individual τ/frequency/component which allows raw values to be seen easier.
	Uses the same normalization strategy as below for each individual frequency/τ/component.
	Global Max Scaling normalizes the entire heatmap against the maximum and minimum values such that any single value x_i is scaled according to this function:
	x_i=  (x_i-〖(0.99)x〗_min)/(x_max-〖(0.99)x〗_min )
	Minimum values being scaled by 0.99 introduces small error but allows tiny values to be seen above 0 and was used for fitting to log functions better during SAM degradation analyses.
	Frequency Domains uses the same function as above but applies it to three equi-volume sections of the frequency/τ space. 
	High frequency/small τ values are separated from middle and low frequency sections to delineate Warburg/Capacitive/Solution Polarization effects

 
	Select Analysis Type:
	Raw Vars Heatmaps
	Uses global settings you provided to generate heatmaps of the following variables averaged across your replicates: Zmod, Zphz,Zreal, Zimag, Creal, Cimag, Cmod
	Also generates a heatmap of standard deviations across all trials.
	Can have Heatmap Norm Strategy applied to it as well.
	DRT Analysis
	DRT Method
	Non-negative least squares, radial basis function, Bayesian Hilbert Transform or M(RQ) fitting
 
	DRT Mode
	Which portion of the impedance data to analyze. 
	Most papers I have read use the imaginary portion only to isolate relaxations but some use the complex.
	Lambda Values
	Comma delineated list of lambda values to run DRTs at.
	Lambda is the regularization constant that balances between error and overfitting.
	Values close to 1 will strongly punish fits that produce sharper peaks. This tends to group multiple processes into a wider peak and introduces error.
	Values closer to 0 allow sharper peaks to exist which can resolve processes of similar timescales but is more sensitive to noise. It may separate a single process into smaller peaks if the number of frequency measurements is low. 
	More EIS measurements per decade of frequency = Better Resolution of peaks in DRT
	M(RQ) fit will provide the same results regardless of lambda, alter the length of the (RC) and (RQ) chain instead.
	RBF Type
	Shape of the assumed curves distributions. 
	Applying common shapes such as Gaussian and Cauchy distributions may help resolve similar timescale processes.
	Radial Basis Function and Bayesian Hilbert methods take a long time to compute.
	RBF Shape
	FWHM is a direct measure of the width of the fitting function’s peak, while the factor is a scaling parameter that can be used to adjust the shape and, consequently, the width of the RBF.
	RBF Size
	Keep this value around 0.5 but you can adjust it slightly if your computationally-expensive RBF/BHT method isn’t resolving peaks.
	Fit Penalty
	Lambda punishes the fitting algorithm for predicting steep slopes. This is the first derivative so the Fit Penalty is set to 1.
	If you wish to track changes at higher derivatives, this can be set to higher integer values.
	Number of DRT Attempts
	NNLS is much faster and achieves good results in under 1000 attempts.
	M(RQ) fitting works quickly for smaller chains of (RQ) and (RC) elements. 
	BHT and RBF work slowly so I suggest varying this upwards from 100 once you get a feel for which other parameters resolve your peaks.
	M(RQ) Fitting CDC
	Keep this as ‘R’ unless running M(RQ) method.
	Accepts any length chain of (RC) and (RQ) elements as well as an R in series.
	Fits the supplied CDC to the data and uses the estimated time constants as a basis function for a DRT.
	Include Inductance
	Should values outside the normal Nyquist curve quadrant be considered? This can create fitting issues because the frequency space EIS data does not converge to 0 at infinitely high and low frequencies.
	Most algorithms can handle these edge cases and BHT can even extrapolate beyond the timescales measured.
 
	ECM Fitting
	ECMs should be a list of single-line circuit codes for the equivalent circuits you wish to fit.
	R, C, W, Q for the standard set of components but there are more.
	() denotes parallel and [] denotes series so Randles is R(C[RW]) and R(Q[RW]) for a CPE modified version
	Fit Method
	Mathematical curve fitting algorithm to fit with. Auto typically uses the ideal method but may use different methods for different measurements so perhaps set it to a commonly effective one such as Powell
	Max Evaluations
	How many steps of the chosen fitting algorithm to conduct on each measurement. I have never seen an algorithm take more than 21,000 steps and they rarely need more than 10,000 but most algorithms know when they have converged and will stop calculating.
	Fit Weight
	Have never experimented with adjusting outside auto.
 
	Peak Tracking
	Min/Max Freq for Peak Search
	The higher and lower frequency  bounds of the data to search for a peak. Use the Raw Vars method without day0 to isolate the Cimag and Zphz region you are interested in.
	Cimag typically maximizes when Zreal contributes maximally to Zmod so frequencies denoting Rsol on a Nyquist curve need to be excluded because they also peak similarly as the Rct frequency.
	If very low frequencies are measured, Warburg impedance may create a new peak as a pseudo-capacitive element. Have not encountered that in data so far but functionality may be added later.
	Plays well with day0 function. 
	Not meant to be used with Global Scaling or Frequency Section normalizations. Raw and Per parameter can be individually useful.

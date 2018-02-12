#####################
# CS 181, Spring 2018
# Homework 1, Problem 3
#
##################

import matplotlib.pyplot as plt
import numpy as np


#############################
######### PART A ############
#############################
# Define a function that will generate n data points for a given K. This will be useful later
def generate_data_points(K,n):
    # Randomly select parameters a_mu, or coefficients of the polynomial,
    # and x_i, points to be sampled
    a_mu = np.random.uniform(-1,1,K+1)
    x_i = np.random.uniform(-5,5,n)
    # Define the polynomial based on the randomly selected parameters. 
    f = lambda x: np.array([a_mu[i]*x**i for i in range(K+1)]).sum(0)
    # Calculate the range of f over the sampled x_i, and set the noise variance
    sigma = ((f(x_i).max()-f(x_i).min())/10.)
    # or sigma = ((f(x_i).max()-f(x_i).min())/10.)**.5
    eta_i = np.random.normal(0,sigma,n)
    # Compute y_i corresponding to x_i
    y_i = f(x_i) + eta_i
    return x_i, y_i, sigma

#Initialize K and n - the degree of the polynomial and the number of data points to be sampled.
K = 10
n = 20

# Call the function to generate the points
x_i,y_i,sigma = generate_data_points(K,n)


#############################
######### PART B ############
#############################
# We can use np.polyfit which minimizes the chi^2.
# Note that the returned chi2 is scaled by sigma^2 based on our defintion
def get_chi2_min(x,y,K,sigma):
    [_,chi2, _, _, _] = np.polyfit(x,y,K,full = True)
    # If we try to fit a polynomial of higher degree than the number of data points,
    # return 0 because we can fit it perfectly
    if K>len(x) or len(chi2)==0:
        return 0.0
    else:
        return chi2[0]/sigma**2.

# Plot chi2min for a range of K (in this case, 2 through 19).
# Using the x_i and y_i generated in part a)
plt.plot(range(2,19),np.array([get_chi2_min(x_i,y_i,R,sigma) for R in range(2,19)]))
plt.title('Chi2 vs. K')
plt.show()



#############################
######### PART C ############
#############################

def optimal_k_trials(n,K,trials):
    optimal_k = np.zeros(trials)
    for i in range(trials):
        x_i,y_i,sigma = generate_data_points(K,n)
        #we need a range of potential k to explore. Here we set it at (2,30] 
        chi2_k = np.array([get_chi2_min(x_i,y_i,k,sigma) for k in range(2,30)])
        BIC = np.array([0.5*n*np.log(2.*np.pi*sigma) - n*np.log(1./10.) + 0.5*(chi2_k[k-2] + (k+1)*np.log(n)) for k in range(2,30)])
        optimal_k[i] = np.argmin(BIC) + 2
    return optimal_k.mean(), optimal_k.std()

n = 20
K = 10
trials = 500
kmean, kstd = optimal_k_trials(n,K,trials)


#############################
######### PART D ############
#############################

# Set values of n that will be computed. Make sure its an int
n_s = np.round(3*np.logspace(0,4,40)).astype(np.int)
# Initialize arrays of mean and variance of optimal k for plotting
kopt_mean = np.zeros(40)
kopt_std = np.zeros(40)
for s in range(40):
    kmean, kstd = optimal_k_trials(n_s[s],K,trials)
    kopt_mean[s] = kmean
    kopt_std[s] = kstd

ax = plt.axes()
ax.set_xscale("log")
plt.errorbar(n_s, kopt_mean, yerr=kopt_std)
plt.xlabel('Number of sample data points (log)')
plt.ylabel('Optimal K (Mean and Std. Deviation)')
plt.show()




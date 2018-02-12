#####################
# CS 181, Spring 2018
# Homework 1, Problem 2
#
##################

import csv
import numpy as np
import matplotlib.pyplot as plt
import sys

csv_filename = 'data/year-sunspots-republicans.csv'
years  = []
republican_counts = []
sunspot_counts = []

with open(csv_filename, 'r') as csv_fh:

    # Parse as a CSV file.
    reader = csv.reader(csv_fh)

    # Skip the header line.
    next(reader, None)

    # Loop over the file.
    for row in reader:

        # Store the data.
        years.append(float(row[0]))
        sunspot_counts.append(float(row[1]))
        republican_counts.append(float(row[2]))

# Turn the data into numpy arrays.
years  = np.array(years)
republican_counts = np.array(republican_counts)
sunspot_counts = np.array(sunspot_counts)
min_sunspots = min(sunspot_counts)
max_sunspots = max(sunspot_counts)
last_year = 1985

# Plot the data.
plt.figure(1)
plt.plot(years, republican_counts, 'o')
plt.xlabel("Year")
plt.ylabel("Number of Republicans in Congress")
plt.figure(2)
plt.plot(years, sunspot_counts, 'o')
plt.xlabel("Year")
plt.ylabel("Number of Sunspots")
plt.figure(3)
plt.plot(sunspot_counts[years<last_year], republican_counts[years<last_year], 'o')
plt.xlabel("Number of Sunspots")
plt.ylabel("Number of Republicans in Congress")
plt.show()

# Create the simplest basis, with just the time and an offset.
X = np.vstack((np.ones(years.shape), years)).T

#################################
# SOLUTION: basis functions

def exp_kernel(x,mu):
    return np.exp(-1/float(25)*np.power(x-mu,2))

def make_basis(xx,part='a'):

    X = [np.ones(xx.shape).T]
    
    if part=='a':
        for j in range(1, 6):
            X.append(xx**j)
    elif part=='b':
        for mu_j in range(1960,2015,5):
            X.append(exp_kernel(xx,mu_j))
    elif part=='c':
        for j in range(1,6):
            X.append(np.cos(xx/float(j)))
    else:
        for j in range(1,26):
            X.append(np.cos(xx/float(j)))

    return np.vstack(X).T


# Nothing fancy for outputs.
Y = republican_counts

# Find the regression weights using the Moore-Penrose pseudoinverse.
def find_weights(X,Y):
    w = np.linalg.solve(np.dot(X.T, X) , np.dot(X.T, Y))
    return w

# Compute the regression line on a grid of inputs.
# DO NOT CHANGE grid_years!!!!!
grid_years = np.linspace(1960, 2005, 200)

for part in ['a','b','c','d']:

    X = make_basis(years,part)   
    w = find_weights(X,Y)

    grid_X = make_basis(grid_years,part)
    grid_Yhat = np.dot(grid_X,w)

    print ("Quadratic Loss, part("+part+"):", np.sum(np.power(Y-np.dot(X,w),2)))

    plt.plot(years, republican_counts,'o',grid_years,grid_Yhat,'-')
    plt.xlabel("Year")
    plt.ylabel("Number of Republicans in Congress")
    plt.show()

grid_sunspots = np.linspace(min_sunspots, max_sunspots, 200)
Y = republican_counts[years<last_year]

for part in ['a','c','d']:

    X = make_basis(sunspot_counts[years<last_year],part)   
    w = find_weights(X,Y)

    grid_X = make_basis(grid_sunspots,part)
    grid_Yhat = np.dot(grid_X,w)

    print ("Quadratic Loss, part("+part+"):", np.sum(np.power(Y-np.dot(X,w),2)))

    plt.plot(sunspot_counts[years<last_year], Y,'o',grid_sunspots,grid_Yhat,'-')
    plt.xlabel("Sunspots")
    plt.ylabel("Number of Republicans in Congress")
    plt.show()

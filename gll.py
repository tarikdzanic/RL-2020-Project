from math import sqrt
import numpy as np

def getGLpoints(nPoints):
	if (nPoints == 1):
		GLpoints = np.array([0.])
	elif (nPoints == 2):
		GLpoints = np.array([-1., 1.])
	elif (nPoints == 3):
		GLpoints = np.array([-1., 0., 1.])
	elif (nPoints == 4):
		GLpoints = np.array([-1., -sqrt(1.0/5), sqrt(1.0/5), 1.])
	elif (nPoints == 5):
		GLpoints = np.array([-1., -sqrt(3.0/7), 0., sqrt(3.0/7), 1.])
	elif (nPoints == 6):
		GLpoints = np.array([-1., -sqrt((1.0/3) + (2*sqrt(7.0)/21.0)), -sqrt((1.0/3) - (2*sqrt(7.0)/21.0)), 
		sqrt((1.0/3) - (2*sqrt(7.0)/21.0)), sqrt((1.0/3) + (2*sqrt(7.0)/21.0)), 1.])
	elif (nPoints == 7):
		GLpoints = np.array([-1., -sqrt((5.0/11) + (2*sqrt(5.0/3)/11.0)), -sqrt((5.0/11) - (2*sqrt(5.0/3)/11.0)), 
		0., sqrt((5.0/11) - (2*sqrt(5.0/3)/11.0)), sqrt((5.0/11) + (2*sqrt(5.0/3)/11.0)), 1.])
	else:
		raise ValueError("Error: No suitable GLL points for nPoints = " + str(nPoints))
	return GLpoints

def getGLweights(nPoints):
	if (nPoints == 1):	
		GLweights = np.array([2.])
	elif (nPoints == 2):
		GLweights = np.array([1., 1.])
	elif (nPoints == 3):		
		GLweights = np.array([1/3.0, 4/3.0, 1/3.0])
	elif (nPoints == 4):	
		GLweights = np.array([1/6.0, 5/6.0, 5/6.0, 1/6.0])
	elif (nPoints == 5):	
		GLweights = np.array([1/10.0, 49/90.0, 32/45.0, 49/90.0, 1/10.0])
	elif (nPoints == 6):	
		GLweights = np.array([1/15.0, (14.0 - sqrt(7.0))/30.0, (14.0 + sqrt(7.0))/30.0,
		(14.0 + sqrt(7.0))/30.0, (14.0 - sqrt(7.0))/30.0, 1/15.0])
	elif (nPoints == 7):
		GLweights = np.array([1/21.0, (124 - 7*sqrt(15))/350.0, (124 + 7*sqrt(15))/350.0, 256/525.0,
		(124 + 7*sqrt(15))/350.0, (124 - 7*sqrt(15))/350.0, 1/21.0])
	else:
		raise ValueError("Error: No suitable GLL weights for nPoints = " + str(nPoints))
	return GLweights


# Computes vector of 1st derivative of Lagrangian l_j functions (l1', l2', etc.) at the ind GLL point (x)
def getGLderivatives(nPoints,ind):
	GLpoints = getGLpoints(nPoints)
	GLderivs = np.zeros(nPoints)
	lsum = 0.0
	lprod = 0.0
	x = GLpoints[ind]

	for j in range(0,nPoints):
		x_j = GLpoints[j]
		lsum = 0.0
		for i in range(0,nPoints):
			lprod = 1.0
			if (i != j):
				for m in range(0,nPoints):
					if (m != i and m != j):
						x_m = GLpoints[m]
						lprod *= (x - x_m)/(x_j - x_m)
				x_i = GLpoints[i]
				lsum += lprod/(x_j - x_i)
		if (abs(lsum) > 1E-12):
			GLderivs[j] = lsum
		else:
			GLderivs[j] = 0.

	return GLderivs

def getGLderivativesAtX(nPoints,x):
	GLpoints = getGLpoints(nPoints)
	GLderivs = np.zeros(nPoints)
	lsum = 0.0
	lprod = 0.0

	for j in range(0,nPoints):
		x_j = GLpoints[j]
		lsum = 0.0
		for i in range(0,nPoints):
			lprod = 1.0
			if (i != j):
				for m in range(0,nPoints):
					if (m != i and m != j):
						x_m = GLpoints[m]
						lprod *= (x - x_m)/(x_j - x_m)
				x_i = GLpoints[i]
				lsum += lprod/(x_j - x_i)
		if (abs(lsum) > 1E-12):
			GLderivs[j] = lsum
		else:
			GLderivs[j] = 0.

	return GLderivs

# Makes matrix of Lagrangian function derivatives at GLL points. M(i,j) = Derivative of l_i at GLL point j
def getGLDerivativeMatrix(nPoints):
	GLDerivMatrix = np.zeros((nPoints, nPoints))
	for j in range(0,nPoints):
		GLDerivMatrix[:,j] = getGLderivatives(nPoints,j)
	return GLDerivMatrix

def projectPoint(x, splitloc, side):
	rat = (x + 1.0)/2.0
	if (side == 'left'):
		return (-1.0 + (splitloc + 1.0)*rat)
	elif (side == 'right'):
		return (splitloc + (1.0 - splitloc)*rat)


def getGLSplitDerivativeMatrix(nPoints, splitloc):
	GLpoints = getGLpoints(nPoints)
	GLDM1 = np.zeros((nPoints, nPoints))
	GLDM2 = np.zeros((nPoints, nPoints))

	for j in range(0,nPoints):
		x = projectPoint(GLpoints[j], splitloc, 'left')
		GLDM1[:,j] = getGLderivativesAtX(nPoints,x)	
	for j in range(0,nPoints):
		x = projectPoint(GLpoints[j], splitloc, 'right')
		GLDM2[:,j] = getGLderivativesAtX(nPoints,x)

	return [GLDM1, GLDM2]

def evalLagPolyAtX(values,x):
	nPoints = len(values)
	GLpoints = getGLpoints(nPoints)

	L = 0.0

	for j in range(0,nPoints):
		l_j = 1.0
		x_j = GLpoints[j]
		for m in range(0,nPoints):
			x_m = GLpoints[m]
			if (m != j):
				l_j *= (x - x_m)/(x_j - x_m)
		L += values[j]*l_j
	return L



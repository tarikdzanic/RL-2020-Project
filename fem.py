import numpy as np
import gll 
import math
import matplotlib.pyplot as plt
import csv
import pickle
import copy


def calculateMassMatrix(order, J):
	w = gll.getGLweights(order+1)
	M = w*J
	return M

def calculateRHS(evalB, order, x, J):
	w = gll.getGLweights(order+1)
	b = evalB(x)
	B = w*b/J
	return -B


def calculateVolumeIntegral(order, J):
	mu = 1.0
	nPoints = (order+1)
	w = gll.getGLweights(nPoints)
	V = np.zeros((nPoints, nPoints))
	D = gll.getGLDerivativeMatrix(nPoints)	

	#V = w*D.T
	for i in range(nPoints):
		for j in range(nPoints):
			for k in range(nPoints):
				V[i,j] += mu*w[k]*D[i,k]*D[j,k]*J
	return V


def makeMesh(nx, var, dom):
	x = np.zeros(nx)
	x[0] = 0
	for i in range(1, nx):
		x[i] = x[i-1] + 0.5*var*np.random.rand() + 1
	x /= x[-1]
	x = dom[0] + (dom[1] - dom[0])*x
	return x


def makeHMesh(nx, var, dom, p):
	x = np.zeros(nx)
	x[0] = 0
	psum = np.sum(p)
	for i in range(1, nx):
		x[i] = x[i-1] + (p[i-1]/psum) + 0.5*var*np.random.rand() 
	x /= x[-1]
	x = dom[0] + (dom[1] - dom[0])*x
	return x

def makeGlobalMatrices(evalB, x, p, lbc, rbc):
	nelems = len(x)-1
	npts = int(np.sum(p+1) - (len(p)-1))
	M = np.zeros(npts)
	Minv = np.zeros(npts)
	V = np.zeros((npts, npts))
	B = np.zeros(npts)
	idx = 0

	for k in range(nelems):
		J = 2.0/(x[k+1] - x[k])
		xi = x[k] + (x[k+1] - x[k])*(gll.getGLpoints(p[k]+1)+1.0)/2.0
		Me = calculateMassMatrix(p[k], J)
		Ve = calculateVolumeIntegral(p[k], J)
		Be = calculateRHS(evalB, p[k], xi, J)
		#print(np.shape(Minv[idx:idx+p[k]+1, idx:idx+p[k]+1]), np.shape(Mie))
		#print(idx,idx+p[k]+1)
		M[idx:idx+p[k]+1] += Me
		V[idx:idx+p[k]+1, idx:idx+p[k]+1]   += Ve
		B[idx:idx+p[k]+1] += Be
		idx += p[k] 

	Minv = 1.0/M
	S = Minv[:,np.newaxis]*V
	B = Minv*B
	S[0, :]  *= 0
	S[-1, :] *= 0
	S[0,0] = 1
	S[-1,-1] = 1
	B[0] = lbc
	B[-1] = rbc

	return [S, B]

def getGlobalX(x,p):
	nelems = len(x)-1
	xout = np.array([])
	for k in range(nelems):
		xi = x[k] + (x[k+1] - x[k])*(gll.getGLpoints(p[k]+1)+1.0)/2.0
		if k != 0:
			xout = np.concatenate((xout,xi[1:]))
		else:
			xout = np.concatenate((xout,xi))
	return xout

def getError(x, u):
	n = len(u)
	uex = exactSol(x)
	err = np.sum(np.square(uex - u))/n
	return (np.sqrt(err))

def makeSystem(dom, lbc, rbc):
	k = []
	c = []
	k = [4.2, 1.9, 8.9]
	f = [0,0,0]
	c = [-1, 3, -2]
	nk = len(k)
	#for i in range(nk):
	#	k.append((np.random.rand()*60))
	#	c.append(((np.random.rand()-0.5)))

	def evalB(x):
		out = 0
		for i in range(nk):
			out += c[i]*np.sin(k[i]*np.pi*x + f[i])
		return out

	def evalf(x):
		out = 0
		for i in range(nk):
			out += -c[i]*np.sin(k[i]*np.pi*x + f[i])/(k[i]*np.pi)**2
		return out

	def exactSol(x):
		lval = evalf(dom[0])
		rval = evalf(dom[1])
		return evalf(x) - lval + lbc  - (rval - rbc - lval + lbc)*(x - dom[0])/(dom[1] - dom[0])
	return [evalB, exactSol]

def getLocalSolution(evalB, exactSol, x,u,p,nint,nidx):
	nelems = len(p)
	idx = 0
	for k in range(nelems):
		if k == nidx:
			xx = x[idx:idx+p[k]+1]
			uu = u[idx:idx+p[k]+1] 
			if k == 0:
				xi = np.linspace(xx[1], xx[-1], nint)
			elif k == nelems -1:
				xi = np.linspace(xx[0], xx[-2], nint)
			else:
				xi = np.linspace(xx[0], xx[-1], nint)
			upoly = np.poly1d(np.polyfit(xx, uu, p[k]))
			ui = upoly(xi)
			#err = getError(xi, ui)
			err = getFunctionalError(evalB, exactSol, upoly, xx)
			return [xi, ui, err]
		else:
			idx += p[k]

	return 

def getFunctionalError(evalB, exactSol, upoly, xx):
	mu = 1.0
	n = 100
	xi = np.linspace(xx[0], xx[-1], n)
	d2u = np.polyder(np.polyder(upoly))
	err = mu*d2u(xi) - evalB(xi)
	err = np.linalg.norm(err)
	return err


def projectSolution(x,u,p,xproj):
	nelems = len(p)
	xgll = xproj
	xout = np.array([])
	uout = np.array([])
	idx = 0
	for k in range(nelems):
		xx = x[idx:idx+p[k]+1]
		uu = u[idx:idx+p[k]+1] 
		upoly = np.poly1d(np.polyfit(xx, uu, p[k]))
		xi = xx[0] + (xx[-1] - xx[0])*xgll
		ui = upoly(xi)
		if k == 0:
			xout = np.concatenate((xout, xi))
			uout = np.concatenate((uout, ui))
		else:
			xout = np.concatenate((xout, xi[1:]))
			uout = np.concatenate((uout, ui[1:]))
		idx += p[k] 
	return [xout, uout]

def getTotalError(x,u,p,ptarg, exactSol):
	xpts = getGlobalX(x,p)
	xproj = np.linspace(0,1,ptarg)
	[xi, ui] = projectSolution(xpts,u,p,xproj)
	uex = exactSol(xi)
	return np.linalg.norm(uex - ui)


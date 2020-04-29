import pickle
import matplotlib.pyplot as plt
import numpy as np
import fem 

def getSolution(state):
	global mesh
	global evalB
	[S, B] = fem.makeGlobalMatrices(evalB, mesh, state, 0,0)
	u = np.linalg.solve(S,B)
	xproj = np.linspace(0,1,20)
	return fem.projectSolution(fem.getGlobalX(mesh, state),u,state,xproj)


i = 9980
with open('dqn_{0}.pickle'.format(i), mode='rb') as f:
	[_, _, data] = pickle.load(f)

[_, _, state, _] = data[-1]

[evalB, exactSol] = fem.makeSystem([0,1], 0, 0)

refstate = np.array([3,3,3,3,3])
mesh = fem.makeHMesh(6, 0.0, [0,1], refstate)
[xref,uref] = getSolution(refstate)
mesh = fem.makeHMesh(6, 0.0, [0,1], state)
[x,u] = getSolution(state)
plt.plot(xref,uref, 'g-')
plt.plot(x,u, 'b--')
plt.plot(x,exactSol(x), 'r-')
plt.legend(['Numerical', 'RL-AMR', 'Analytic'])
plt.show()
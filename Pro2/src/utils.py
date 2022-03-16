import pypoman
from numpy import array, eye, ones, vstack, zeros
import numpy as np
import scipy
import pdb
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull


class system(object):
	"""docstring for system"""
	def __init__(self, A, B, x0):
		self.A     = A
		self.B     = B
		self.x 	   = [x0]
		self.u 	   = []
		self.w 	   = []
		self.x0    = x0
			
	def applyInput(self, ut):
		self.u.append(ut)
		xnext = np.dot(self.A,self.x[-1]) + np.dot(self.B,self.u[-1])
		self.x.append(xnext)

	def reset_IC(self):
		self.x = [self.x0]
		self.u = []
		self.w = []

def dlqr(A, B, Q, R):
	# solve the ricatti equation
	P = np.matrix(scipy.linalg.solve_discrete_are(A, B, Q, R))
	# compute the LQR gain
	K   = np.array(scipy.linalg.inv(B.T*P*B+R)*(B.T*P*A))
	Acl = A - np.dot(B, K)
	return P, K, Acl
import pypoman
from numpy import array, eye, ones, vstack, zeros
import numpy as np
import scipy
import pdb
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

class polytope(object):
	"""docstring for polytope"""
	def __init__(self, F=None, b=None, vertices=None):

		if vertices == None:
			self.b = b
			self.F = F
			self.computeVRep()
		else:
			self.vertices = vertices
			self.computeHRep()


	def NStepPreAB(self, A, B, Fu, bu, N):
		F = self.F
		b = self.b

		for i in range(0, N):
			FPreAB, bPreAB = self.preAB(A, B, Fu, bu)
			self.F = FPreAB
			self.b = bPreAB
					
		F_NStepPreAB = self.F
		b_NStepPreAB = self.b
		
		self.F = F
		self.b = b

		return F_NStepPreAB, b_NStepPreAB

	def computeO_inf(self, A):
		for i in range(0,5):
			Fpre, bpre = self.preA(A)
			self.intersect(Fpre, bpre)
		return self.F, self.b
	
	def computeC_inf(self, A, B):
		for i in range(0,5):
			Fpre, bpre = self.preAB(A, B, self.F.shape[1])
			self.intersect(Fpre, bpre)

	def preA(self, A):
		b = self.b
		F = np.dot(self.F, A)	
		return F, b
	
	def intersect(self, F_intersect, b_intersect):
		self.F = np.vstack((self.F, F_intersect))
		self.b = np.hstack((self.b, b_intersect))
		
	def computeVRep(self):
		self.vertices = pypoman.duality.compute_polytope_vertices(self.F, self.b)

	def computeHRep(self):
		self.F, self.b = pypoman.duality.compute_polytope_halfspaces(self.vertices)

	def preAB(self, A, B, Fu, bu):
		n = A.shape[1] 
		d = B.shape[1]

		# Original polytope:
		F1 = np.hstack( ( np.dot(self.F, A), np.dot(self.F, B) ) )
		b1 = self.b
		
		F2 = np.hstack( ( np.zeros((Fu.shape[0], n)), Fu ) )
		b2 = bu
		ineq = (np.vstack((F1, F2)), np.hstack(( b1, b2 )) )  # A * x + Bu <= b, F_u u <= bu 

		# Projection is proj(x) = [x_0 x_1]
		E          = np.zeros(( n, n+d ))
		E[0:n,0:n] = np.eye(n)
		f          = np.zeros(n)
		proj       = (E, f)  # proj(x) = E * x + f

		vertices = pypoman.project_polytope(proj, ineq)#, eq=None, method='bretl')
		F, b = pypoman.duality.compute_polytope_halfspaces(vertices)
		return F, b

	def plot2DPolytope(self, color, label = None):
		# This works only in 2D!!!!
		vertices  = pypoman.polygon.compute_polygon_hull(self.F, self.b)
		vertices.append(vertices[0])
		xs, ys = zip(*vertices) #create lists of x and y values
		plt.plot(xs, ys, '-o', color=color, label=label)
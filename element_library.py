import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as mplanim
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d as a3
import time as time
import timeit



class FiniteElement:
    ndims = 2
    def __init__(self, E=210000, nu=0.3, thickness=1):
        self.nodes = list()
        self.dofs = None  # Assigned on creation

        self.thickness = thickness
        self.C = E/(1-nu**2) * np.array([   [1, nu, 0],
                                            [nu, 1, 0],
                                            [0, 0, (1-nu)/2]])

    def stiffness_matrix(self):
        integration_points, weights = self.f_integration_points()

        k = np.zeros(2 * (self.ndims*self.r.shape[0], ))

        for r,w in zip(integration_points, weights):
            B = self.B(*r)
            k = k + B.T @ self.C @ B * np.linalg.det(self.jacobian(*r))
        if self.ndims == 2:
            return k*self.thickness
        elif self.ndims == 3:
            return k

    def f_integration_points(self):
        """
        :return: tuple (weights, integration_points)
        """
        if self.__class__.__name__[0:3] == 'Tri':
            a = np.sqrt(3/5)
            integration_points = np.array([[-a, a], [0, a], [a, a],
                                           [-a, 0], [0, 0], [a, 0],
                                           [-a, -a], [0, -a], [a, -a]])
            weights = 1/81 * np.array([0, 15, 3.125,
                                       0, 16, 15,
                                       0, 0, 0])

        elif self.n_integration_points == 1:
            integration_points = np.array([[0,0]])
            weights = np.array([1])

        elif self.n_integration_points == 2:
            a = 1/np.sqrt(3)
            integration_points = np.array([[ -a, a], [ a, a],
                                           [ -a,-a], [ a,-a]])
            weights = np.array([1, 1, 1, 1])

        elif self.n_integration_points == 3:
            a = np.sqrt(3/5)
            integration_points = np.array([[-a, a],  [0, a],  [a, a],
                                           [-a, 0],  [0, 0],  [a, 0],
                                           [-a, -a], [0, -a], [a,-a]])
            weights = 1/81 * np.array([25, 40, 25,
                                       40, 64, 40,
                                       25, 40, 25])
            # weights = np.outer([1, 5/8, 1], [1, 5/8, 1]) * 25/81 .flatten()
        return integration_points, weights

    def jacobian(self, *args):
        return self.G(*args) @ self.r

    def B(self, *args):
        if len(args) == 2:
            dN_dx, dN_dy = np.linalg.solve(self.jacobian(*args) , self.G(*args))
            return np.array((np.array((dN_dx, np.zeros_like(dN_dx))).T.flatten(),
                             np.array((np.zeros_like(dN_dy), dN_dy)).T.flatten(),
                             np.array((dN_dy, dN_dx)).T.flatten()
                             ))
        elif len(args) == 3:
            dN_dx, dN_dy, dN_dz = np.linalg.solve(self.jacobian(*args), self.G(*args))
            zeroslike = np.zeros_like(dN_dx)
            return np.array((np.array((dN_dx, zeroslike, zeroslike)).T.flatten(),
                             np.array((zeroslike, dN_dy, zeroslike)).T.flatten(),
                             np.array((zeroslike, zeroslike, dN_dz)).T.flatten(),
                             np.array((dN_dy, dN_dx, zeroslike)).T.flatten(),
                             np.array((dN_dz, zeroslike, dN_dx)).T.flatten(),
                             np.array((zeroslike, dN_dz, dN_dy)).T.flatten()
                             ))

    def G(self, *args):
        """
        This function outputs the derivatives of the shape functions as a 2xN array.
        Possibly with some abuse of notation, it can be written as the outer product (2D)
        [d/dxi d/deta].T @ [N1 N2 ... Nn]

        For some execution speed gains, subclasses of FiniteElement can implement
        their own G() with partial derivatives programmed directly.

        :return:
        """
        SS = self.shape_functions(*args)
        if len(args) == 2:
            dN_dxi = (self.shape_functions(args[0] + 1e-5, args[1]) - SS)/1e-5
            dN_deta = (self.shape_functions(args[0], args[1] + 1e-5) - SS)/1e-5
            return np.array((dN_dxi, dN_deta))
        elif len(args) == 3:
            dN_dxi = (self.shape_functions(args[0] + 1e-5, args[1], args[2]) - SS) / 1e-5
            dN_deta = (self.shape_functions(args[0], args[1] + 1e-5, args[2]) - SS) / 1e-5
            dN_dzeta = (self.shape_functions(args[0], args[1], args[2] + 1e-5) - SS) / 1e-5
            return np.array((dN_dxi, dN_deta, dN_dzeta))

    def displacements(self):
        return np.array([node.displacements for node in self.nodes])

    def displacements_(self, *args):
        """
        !! May be broken for 3D problems
        Solution: [N1, 0, 0, N2, 0, 0]
                  [0, N1, 0, 0, N2, 0]
                  [0, 0, N3, 0, 0, N3] ?
        :param args:
        :return:
        """
        N = self.shape_functions(*args)
        return np.array((np.array((N, np.zeros_like(N))).T.flatten(),
                         np.array((np.zeros_like(N), N)).T.flatten()
                         )) @ self.displacements().flatten()

    def strain(self, *args):
        return self.B(*args) @ self.displacements().flatten()

    def stress(self, *args):
        return self.C @ self.strain(*args)

    def stress_vonmises(self, *args):
        sxx, syy, sxy = self.stress(*args)
        return np.sqrt(sxx**2 - sxx*syy + syy**2 + 3*sxy**2)

    def Ex(self, newdim):
        dofs = np.array([node.dofs for node in self.nodes]).flatten()
        E = np.zeros((newdim, len(dofs)))
        for i, j in enumerate(dofs):
            E[j, i] = 1
        return E

    def __eq__(self, other):
        return np.allclose(self.r, other.r)

class FiniteElement3D(FiniteElement):
    ndims = 3
    def __init__(self, E=210000, nu=0.3):
        self.nodes = list()
        self.dofs = None

        self.C = E/((1+nu)*(1-2*nu)) * np.array([[ 1-nu, nu, nu, 0, 0, 0 ],
                                                 [ nu, 1-nu, nu, 0, 0, 0 ],
                                                 [ nu, nu, 1-nu, 0, 0, 0 ],
                                                 [0, 0, 0, 0.5-nu, 0, 0 ],
                                                 [0, 0, 0, 0, 0.5-nu, 0 ],
                                                 [0, 0, 0, 0, 0, 0.5-nu]   ])

    def f_integration_points(self):
        """
        :return: tuple (integration_points, weights)
        """

        if self.n_integration_points == 2:
            a = 1 / np.sqrt(3)
            integration_points = np.array([[-a, a, -a], [a, a, -a],
                                           [-a, -a, -a], [a, -a, -a],
                                           [-a, a, a], [a, a, a],
                                           [-a, -a, a], [a, -a, a],
                                           ])
            weights = np.ones(8)

        return integration_points, weights

class Node:
    def __init__(self, r, ndofs):
        self.r = np.asarray(r)

        self.elements = list()

        self.ndofs = ndofs
        self.loads = np.zeros(ndofs)
        self.displacements = np.zeros(ndofs)

        self.id = None
        self.dofs = None

    def add_element(self, element):
        self.elements.append(element)

    def __eq__(self, other):
        return np.allclose(self.r, other.r)

class Node2D(Node):
    def __init__(self, r):
        super().__init__(r, ndofs=2)

class Node3D(Node):
    def __init__(self, r):
        super().__init__(r, ndofs=3)
# As Node is initialized with argument ndofs, Node2D and Node3D may be unneeded

class Quad4(FiniteElement):
    n_integration_points = 2
    plotidx = [0, 1, 2, 3]
    def __init__(self, r, *args, **kwargs):
        """
        :param r:  4x2 array of coordinates: [[x1,y1], [x2, y2], .. ]
        
        In (xi, eta): 
        1: (-1,-1) | N1 = 1/4 (1-xi)(1-eta) 
        2: (1,-1)  | N2 = 1/4 (1+xi)(1-eta) 
        3: (1,1)   | N3 = 1/4 (1+xi)(1+eta) 
        4: (-1,1)  | N4 = 1/4 (1-xi)(1+eta)
        """

        super().__init__(**kwargs)

        self.r = r
        #self.n_integration_points = 2

    def shape_functions(self, *args):
        xi, eta = args[0:2]
        return 1/4 * np.array([(1-xi)*(1-eta),
                               (1+xi)*(1-eta),
                               (1+xi)*(1+eta),
                               (1-xi)*(1+eta)])

    def G(self, *args):
        xi, eta = args[0:2]
        return 1/4 * np.array([[ -(1-eta), 1-eta,  1+eta, -(1+eta)],
                               [ -(1-xi), -(1+xi), 1+xi,   1-xi]])

class Quad4R(Quad4):
    n_integration_points = 1
    def __init__(self, r, *args, **kwargs):
        super().__init__(r, *args, **kwargs)

class Quad8(FiniteElement):
    n_integration_points = 3
    plotidx = [0, 4, 1, 5, 2, 6, 3, 7]
    def __init__(self, r, *args, **kwargs):
        """
                :param r:  8x2 array of coordinates: [[x1,y1], [x2, y2], .. ]
                If a 4x2 array of coordinates is passed, will linearly interpolate midnodes between vertex nodes
                Nodes 5, 6, 7, 8 are midside nodes
                """
        super().__init__(*args, **kwargs)

        self.r = r
        if self.r.shape == (4,2):
            self.r = np.array([*r,
                               (r[0]+r[1])/2,
                               (r[1]+r[2])/2,
                               (r[2]+r[3])/2,
                               (r[3]+r[0])/2])

    def shape_functions(self, *args):
        # Nodes 1-4 are corner nodes and 5-8 are midside nodes
        xi, eta = args[0:2]
        return 1/4 * np.array([(1-xi)*(eta-1)*(xi+eta+1),
                               (1+xi)*(eta-1)*(eta-xi+1),
                               (1+xi)*(1+eta)*(xi+eta-1),
                               (xi-1)*(eta+1)*(xi-eta+1),
                               2*(1-eta)*(1-xi**2),
                               2*(1+xi)*(1-eta**2),
                               2*(1+eta)*(1-xi**2),
                               2*(1-xi)*(1-eta**2)])

    def G(self, *args):
        xi, eta = args[0:2]
        return 1/4 * np.array([ [
                                    (-eta + 1)*(eta + xi + 1) + (eta - 1)*(-xi + 1),
                                    (eta - 1)*(eta - xi + 1) - (eta - 1)*(xi + 1),
                                    (eta + 1)*(xi + 1) + (eta + 1)*(eta + xi - 1),
                                    (eta + 1)*(xi - 1) + (eta + 1)*(-eta + xi + 1),
                                    -4*xi*(-eta + 1),
                                    -2*eta**2 + 2,
                                    -4*xi*(eta + 1),
                                    2*eta**2 - 2
                                ],[
                                    (eta - 1)*(-xi + 1) + (-xi + 1)*(eta + xi + 1),
                                    (eta - 1)*(xi + 1) + (xi + 1)*(eta - xi + 1),
                                    (eta + 1)*(xi + 1) + (xi + 1)*(eta + xi - 1),
                                    (-eta - 1)*(xi - 1) + (xi - 1)*(-eta + xi+1),
                                    2*xi**2 - 2,
                                    -4*eta*(xi + 1),
                                    -2*xi**2 + 2,
                                    -4*eta*(-xi + 1)
                               ] ])

class Quad8R(Quad8):
    n_integration_points = 2
    def __init__(self, r, *args, **kwargs):
        super().__init__(r, *args, **kwargs)

class Tri3(FiniteElement):
    def __init__(self, r, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.plotidx = [0,1,2]
        self.r = r

    def _shape_functions(self, *args):
        xi, eta = args[0:2]
        A = 1/2 * np.linalg.det( np.block([np.ones(3).reshape((3,1)), self.r]) )
        (x1,y1),(x2,y2),(x3,y3) = self.r
        a1 = x2*y3 - x3*y2
        a2 = x3*y1 - x1*y3
        a3 = x1*y2 - x2*y1
        b1 = y2 - y3
        b2 = y3 - y1
        b3 = y1 - y2
        c1 = x3 - x2
        c2 = x1 - x3
        c3 = x2 - x1

        return 1/(2*A) * np.array([a1 + b1*xi + c1*eta,
                                   a2 + b2*xi + c2*eta,
                                   a3 + b3*xi + c3*eta])

        pass

    def shape_functions(self, *args):
        xi, eta = args[0:2]
        return np.array([eta,
                         xi,
                         1-eta-xi])

class Tri6(FiniteElement):
    def __init__(self, r, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.plotidx = [0,3,1,4,2,5]
        self.r = r
        if self.r.shape == (3,2):
            self.r = np.array([*r,
                               (r[0] + r[1]) / 2,
                               (r[1] + r[2]) / 2,
                               (r[0] + r[2]) / 2])

    def shape_functions(self, *args):
        xi, eta = args[0:2]
        zeta = 1 - xi - eta
        return np.array([xi*(2*xi - 1),
                         eta*(2*eta - 1),
                         zeta*(2*zeta - 1),
                         4*xi*eta,
                         4*eta*zeta,
                         4*zeta*xi])

class Brick8(FiniteElement3D):
    """
    https://www.sharcnet.ca/Software/Abaqus/6.14.2/v6.14/books/stm/default.htm?startat=ch03s02ath62.html
    Abaqus Theory Guide - 3.2.4 Solid isoparametric quadrilaterals and hexahedra
    """
    n_integration_points = 2
    def __init__(self, r, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.r = r

    def shape_functions(self, *args):
        xi, eta, zeta = args[0:3]
        return 1/8 * np.array([ (1-xi)*(1-eta)*(1-zeta),
                                (1+xi)*(1-eta)*(1-zeta),
                                (1+xi)*(1+eta)*(1-zeta),
                                (1-xi)*(1+eta)*(1-zeta),
                                (1-xi)*(1-eta)*(1+zeta),
                                (1+xi)*(1-eta)*(1+zeta),
                                (1+xi)*(1+eta)*(1+zeta),
                                (1-xi)*(1+eta)*(1+zeta)
                              ])

class Problem:
    def __init__(self, ndofs=2):
        self.nodes = list()
        self.elements = list()
        self.ndofs = ndofs

        self.constrained_dofs = np.array([], dtype=int)
        self.loads = np.array([])
        self.displacements = np.array([])
        self.nodal_coordinates = self.f_nodal_coordinates()
        self.nc_dict = dict()

    def create_element(self, r, etype, E=2e5, nu=0.3, *args, **kwargs):
        element = etype(r, E, nu, **kwargs)

        self.elements.append(element)
        element.number = len(self.elements)
        dofs = list()
        for point in element.r:
            node = self.create_node(point)
            node.elements.append(element)
            element.nodes.append(node)
            dofs += node.dofs.tolist()
        element.dofs = np.asarray(dofs)

    def create_node(self, r):
        r = np.asarray(r)
        node = Node(r, ndofs=self.ndofs)
        if tuple(r) not in self.nc_dict:
            self.nodes.append(node)
            self.nc_dict[tuple(r)] = len(self.nodes) - 1
            node.number = len(self.nodes) - 1
            node.dofs = self.ndofs * node.number + np.arange(self.ndofs)
            return node
        else:
            #print('Node already exists at', r)
            return self.nodeobj_at(r)
            pass

    def load_node(self, r, loads):
        self.nodeobj_at(r).loads = np.asarray(loads)
        self.loads = np.array([node.loads for node in self.nodes]).flatten()

    def pin(self, node):
        self.constrained_dofs = np.append(self.constrained_dofs, node.dofs)

    def stiffness_matrix(self, reduced=False):
        self.nodal_coordinates = self.f_nodal_coordinates()
        Ndofs = self.system_ndofs()
        K = np.zeros((Ndofs, Ndofs)) 
        for element in self.elements:
            K[np.ix_(element.dofs, element.dofs)] += element.stiffness_matrix()
        if not reduced:
            return K
        elif reduced:
            K = np.delete(K, self.constrained_dofs, axis=0)
            K = np.delete(K, self.constrained_dofs, axis=1)
            return K

    def free_dofs(self):
        return np.delete(np.arange(self.system_ndofs()), self.constrained_dofs)

    def solve(self):
        free = self.free_dofs()
        Kr = self.stiffness_matrix(reduced=True)

        self.displacements = np.zeros(self.system_ndofs())
        self.displacements[free] = np.linalg.solve(Kr, self.loads[free])
        for node in self.nodes:
            node.displacements = self.displacements[node.dofs]

    def system_ndofs(self):
        return self.ndofs * len(self.nodes) 

    def nodeobj_at(self, r):
        return self.nodes[self.nc_dict[tuple(r)]]

    def f_nodal_coordinates(self):
        return np.array([node.r for node in self.nodes])

    def model_size(self):
        xy = self.nodal_coordinates
        if not np.any(xy):
            return 1
        else:
            model_size = np.sqrt( (np.max(xy[:,0]) - np.min(xy[:,0]))**2 + (np.max(xy[:,1]) - np.min(xy[:,1]))**2)
            return model_size

    def renumber_dofs(self):
        pass

    # Exclusive to Problem2D - May be moved
    def mesh_grid(self, x, y, type=Quad4):
        i = 0
        for x1, x2 in zip(x, x[1:]):
            for y1, y2 in zip(y, y[1:]):
                i += 1
                r = np.array([[x1, y1],
                              [x2, y1],
                              [x2, y2],
                              [x1, y2]])
                if cls.__name__[0:4] == 'Quad':
                    self.create_element(r, etype=cls)
                elif cls == Tri3 or cls == Tri6:
                    xm, ym = 1/2 * np.array([x1+x2, y1+y2])
                    for (xa,ya),(xb,yb) in zip(r, np.roll(r,1,axis=0)):
                        r = np.array([[xa, ya],
                                      [xb, yb],
                                      [xm, ym]])
                        self.create_element(r, etype=cls)

    def plot_prep(self, ncols=1, **kwargs):
        fig,axs = plt.subplots(ncols=ncols, **kwargs)
        axs = (axs,) if ncols==1 else axs   # Make sure axs is iterable
        xmin, xmax = np.min(self.f_nodal_coordinates().T[0]), np.max(self.f_nodal_coordinates().T[0]) + self.model_size() / 5
        ymin, ymax = np.min(self.f_nodal_coordinates().T[1]), np.max(self.f_nodal_coordinates().T[1])

        for ax in axs:
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(xmin, xmax)
            ax.set_aspect('equal')

        return fig,axs

    def plot_model(self, ax):
        for element in self.elements:
            r = element.r[element.plotidx]
            rect = patches.Polygon(r, linewidth=1, facecolor='white', edgecolor='black')
            ax.add_patch(rect)

    def plot_displaced(self, ax, scale_factor=1):
        elemental_stress = np.array([element.stress(0, 0)[0] for element in self.elements])
        maxstress = np.max(elemental_stress) * 1.01
        for element in self.elements:
            r = element.r[element.plotidx] + element.displacements()[element.plotidx] * scale_factor
            polygon = patches.Polygon(r, linewidth=1, facecolor=(np.abs(element.stress(0,0)[0]/maxstress),
                                                                 0, 0),
                                      edgecolor='black')
                                                                 #1 - np.sqrt(element.stress_vonmises(0,0)/maxstress)))
            ax.add_patch(polygon)

    def plot_loads(self, ax):
        scale_factor = 1/2 * self.model_size() / np.linalg.norm(self.loads)

        for node in self.nodes:
            if not np.allclose(node.loads, 0):
                arrow = patches.FancyArrow(*(node.r + node.displacements), *(node.loads * scale_factor),
                                           head_width=self.model_size()/10, facecolor='black',
                                           overhang=0.25)
                ax.add_patch(arrow)

    def plot_functions(self, *args):
        """
        :param args: List of functions, or list of tuples of functions.
        For tuples of functions, all functions in the tuple will be plotted on a single ax
        :return:
        """
        fig, axs = self.plot_prep(ncols=len(args))
        axs = (axs,) if len(args)==1 else axs
        for ax,arg in zip(axs,args):
            arg = (arg, ) if type(arg).__name__ == 'method' else arg  # Make sure arg is iterable
            for func in arg:
                func(ax)

    def animate(self):
        fig,ax = plt.subplots()

        scale_factor = 0
        n_frames = 50
        self.plot_displaced(ax, scale_factor=scale_factor)
        xmin, xmax = np.min(self.f_nodal_coordinates().T[0]), \
                     np.max(self.f_nodal_coordinates().T[0]) + self.model_size() / 5
        def updAnimation(k):
            ax.clear()

            ax.set_xlim(xmin, xmax)
            ax.set_ylim(xmin, xmax)
            ax.set_aspect('equal')
            self.plot_displaced(ax, scale_factor=k/n_frames)

        self.anim = mplanim.FuncAnimation(fig, updAnimation,
                                          frames=n_frames, interval=0.001, blit=False)
        plt.show()

class Problem2D(Problem):
    def __init__(self):
        super().__init__(ndofs=2)

class Problem3D(Problem):
    def __init__(self):
        super().__init__(ndofs=3)

    def mesh_extrude(self, base, extrude_to, aspect_ratio=1):
        """

        :param base: Nx8 np array: each row are corner coordinates for a 4-sided polygon
        Given as [[x1, y1, x2, y2, x3, y3, x4, y4](element 1), [x1, y1, x2, ...](element 2), .. ]
        or as a tuple of 4x2 arrays
        :param aspect_ratio: Approx. length to sqrt(A) ratio of elements in extrude direction
        :return:
        """
        if np.ndim(base) == 2:
            base = (base, )
        for xxyy in base:
            r = xxyy.reshape((4, 2))
            elm_area =  1/2 * sum(  [r[i,0]*r[i+1,1] - r[i+1,0]*r[i,1] for i in range(len(r)-1)] +
                                    [r[-1,0]*r[0,1] - r[0,0]*r[-1,1]]
                                    )
            elm_length = extrude_to / np.round(extrude_to / (aspect_ratio*np.sqrt(elm_area)))
            num_elements = int(extrude_to / elm_length)

            rr = np.vstack((r, r))
            for i in range(num_elements):
                zz = np.hstack((np.zeros(4), np.ones(4) * elm_length)).reshape((8,1)) \
                     + elm_length * i
                r = np.hstack((rr, zz))
                self.create_element(r, Brick8)

    def plot_prep(self, ncols=1, **kwargs):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        xmin, ymin, zmin = np.min(self.f_nodal_coordinates(), axis=0)
        xmax, ymax, zmax = np.max(self.f_nodal_coordinates(), axis=0)

        maxrange = np.array([xmax-xmin, ymax-ymin, zmax-zmin]).max() / 2
        midx = (xmax+xmin)/2
        midy = (ymax+ymin)/2
        midz = (zmax+zmin)/2

        ax.set_xlim(midx-maxrange, midx+maxrange)
        ax.set_ylim(midy-maxrange, midy+maxrange)
        ax.set_zlim(midz-maxrange, midz+maxrange)
        ax.set_aspect('equal')

        return fig,ax

    def plot_model(self):
        fig,ax = self.plot_prep()

        for node in self.nodes:
            pass
            #ax.scatter(*node.r, 'ro')
        for element in self.elements:
            face_polygons = [element.r[k:k+3] for k in range(len(element.r)-3)]
            shape = a3.art3d.Poly3DCollection(face_polygons)
            ax.add_collection3d(shape)

    def plot_as_points(self):
        fig,ax = self.plot_prep()

        for element in self.elements:
            midp = np.average(element.r, axis=0)
            ax.scatter(*midp, color='black')

    def plot_displaced_as_points(self):
        fig,ax = self.plot_prep()

        maxdisp = np.max(p.displacements)

        for element in self.elements:
            midp = np.average(element.r + element.displacements(), axis=0)
            ax.scatter(*midp, color=(element.displacements().max()/maxdisp, 0, 0))

# Problem2D and Problem3D are probably unneeded for the same reason as Node2D/Node3D, but
# may be intuitively useful


def R2(theta):
    s, c = np.sin(theta), np.cos(theta)
    return np.array([[c, -s], [s, c]])

def subdivide_quadrilateral(rr):
    """

    :param rr: Nx2 points
    :param n: Number of subdivisions per direction. Now, two 
    :return:
    """
    r1,r2,r3,r4 = rr
    r5 = (r2 + r1)/2 
    r6 = (r2 + r3)/2 
    r7 = (r3 + r4)/2 
    r8 = (r4 + r1)/2 
    r9 = (r6 + r8)/4 + (r5 + r7)/4 
    
    rr1 = np.vstack((r1, r5, r9, r8))
    rr2 = np.vstack((r5, r2, r6, r9))
    rr3 = np.vstack((r9, r6, r3, r7))
    rr4 = np.vstack((r8, r9, r7, r4))

    return (rr1,rr2,rr3,rr4)

sq = subdivide_quadrilateral



if __name__ == '__main__':

    lc=1

    if lc == 1:
        start_time = time.time()
        r1 = np.array([[0,0],
                      [10,0],
                      [10,10],
                      [0,10]])


        p = Problem()
        n = 8
        x = 10*np.linspace(0,10, 5*n+1)
        y = 10*np.linspace(0,2, 12)
        cls = Quad8
        p.mesh_grid(x, y, type=cls)
        mid_time = time.time()
        p.load_node((100,0), (0,10000))
        p.load_node((100,20), (0,10000))
        #p.load_node((100,20), (-10000, 0))
        #p.load_node((100,0), (10000, 0))
        for _y in y:
            p.pin(p.nodeobj_at((0,_y)))
        q = p.elements[-1]
        n1 = p.nodes[-1]
        p.solve()
        print(q.displacements())
        end_time = time.time()

        q8 = Quad8(r1)
        q4 = Quad4(r1)
        #e1 = p.nodeobj_at((80, 10)).elements[0]
        #e2 = p.nodeobj_at((10, 20)).elements[0]
        #e3 = p.nodeobj_at((10, 0)).elements[0]

        dt = end_time - start_time
        print('n = ', n)
        print('Mesh time', mid_time - start_time)
        print('Assembly and solution time', end_time - mid_time)
        print('Total time', dt)
        filename = '{}_{}.png'.format(cls.__name__, n)
        #p.plot_both(save=False, filename=filename)
        #p.plot_functions(p.plot_model, p.plot_displaced)
        #p.plot()
        p.animate()
        plt.show()

    if lc == 2:
        for n in (2,3,4,5):
            p = Problem()
            #n = 5
            x = 10 * np.linspace(0, 10, 5*n+1)
            y = 10 * np.linspace(0, 2, n+1)

            cls = Tri3
            p.mesh_grid(x, y, type=cls)
            p.load_node((100,0),(0,5000))
            for _y in y:
                p.pin(p.nodeobj_at((0,_y)))
            tri = p.elements[-1]
            t6 = Tri6(np.array([[0,0],[1,0],[0,1]]))

            p.solve()

            print('n = ', n)
            print(tri.displacements())
            filename = '{}_{}_.png'.format(cls.__name__, n)
            p.plot_displaced(save=1, filename=filename)
            #plt.show()
    if lc == 3:
        p = Problem()
        x = np.array([0, 10])
        y = np.array([0, 10])
        p.mesh_grid(x, y, type=Quad8)

        q8 = p.elements[0]
        p.pin(p.nodeobj_at((0,0)))
        p.pin(p.nodeobj_at((0, 5)))
        p.pin(p.nodeobj_at((0, 10)))
        p.load_node((10,0), (1,0))
        p.load_node((10,5), (10000,0))
        p.load_node((10,10), (1,0))
        p.solve()
        p.plot_displaced()
        plt.show()
    if lc == 4:
        p = Problem()
        cls = Quad8

        R = R2(np.deg2rad(30))
        r1 = np.array([[0, 1],
                       [15, 3],
                       [10, 10],
                       [0, 10]]) @ R.T


        p.create_element(r1, etype=cls)

        p.load_node(R@np.array([10,10]), R@np.array([10000,0]))
        #p.pin(p.nodeobj_at((0,1)))
        p.pin(p.nodeobj_at(R@np.array([0,10])))
        p.pin(p.nodeobj_at(R@np.array([0,1])))
        p.solve()
        elm = p.elements[0]

        #p.plot_functions(p.plot_model, (p.plot_displaced, p.plot_loads))
        p.animate(); plt.show()
    if lc == 5:
        start_time = time.time()
        r1 = np.array([[0, 0, 0],
                       [10, 0, 0],
                       [10, 10, 0],
                       [0, 10, 0],
                       [0, 0, 10],
                       [10, 0, 10],
                       [10, 10, 10],
                       [0, 10, 10]])

        p = Problem3D()
        p.create_element(r1, etype=Brick8)
        b = p.elements[0]
        K = p.stiffness_matrix()
    if lc == 6:
        p = Problem3D()
        r0 = np.array([[0,0],[10,0],[10,10],[0,10]])
        r0base = subdivide_quadrilateral(r0)
        p.mesh_extrude(base=r0base, extrude_to=200, aspect_ratio=1)
        for r in r0:
            p.pin(p.nodeobj_at(np.array([*r, 0])))
        for r in p.f_nodal_coordinates()[-4:]:
            p.load_node(r, (0, 1000/4, 0))
        p.solve()
        b = p.elements[int(len(p.elements)/2)]
        print(p.elements[-1].displacements())
        """
        Analytical solution to this problem: 16 mm 
        (1x1)xN elements: 10.3mm 
        (2x2)xN elements: 14.4mm
        """

    p.plot_functions(p.plot_displaced)
    plt.show()





import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as mplanim
import time as time

class Node:
    def __init__(self, xy, draw=False):
        self.x, self.y = xy
        self.r = np.asarray(xy)

        self.elements = list()
        self.loads = np.array([0, 0])  # self.loads (global Fx, Fy) assigned on loading

        self.id = None  # (node id) assigned on creation
        self.dofs = None  # self.dofs (dof1, dof2) assigned on creation
        self.displacements = np.array([0, 0])  # self.displacement (ux, uy) asssigned on solution

    def add_element(self, element):
        self.elements.append(element)

    def __str__(self):
        return '{},{}'.format(self.x, self.y)

    def __eq__(self, other):
        return np.allclose(self.r, other.r)

class FiniteElement:
    def __init__(self, E=210000, nu=0.3, thickness=1):
        self.nodes = list()

        self.thickness = thickness
        self.C = E/(1-nu**2) * np.array([   [1, nu, 0],
                                            [nu, 1, 0],
                                            [0, 0, (1-nu)/2]])

    def stiffness_matrix(self):
        integration_points, weights = self.f_integration_points()

        k = np.zeros(2 * (2*self.r.shape[0], ))

        for r,w in zip(integration_points, weights):
            B = self.B(*r)
            k = k + B.T @ self.C @ B * np.linalg.det(self.jacobian(*r))
        return k*self.thickness
        #k = np.sum(self.B(*r).T @ self.C @ self.B(*r) * self.thickness * np.linalg.det(self.jacobian(*r)) * w
        #       for r,w in zip(integration_points, weights))

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

    def jacobian(self, xi=0, eta=0):
        return self.G(xi, eta) @ self.r

    def B(self, xi=0, eta=0):
        dN_dx, dN_dy = np.linalg.solve(self.jacobian(xi, eta) , self.G(xi, eta))

        return np.array((np.array((dN_dx, np.zeros_like(dN_dx))).T.flatten(),
                         np.array((np.zeros_like(dN_dy), dN_dy)).T.flatten(),
                         np.array((dN_dy, dN_dx)).T.flatten()
                         ))

    def __B(self, xi=0, eta=0):
        dN_dxi = (self.shape_functions(xi+0.01, eta) - self.shape_functions(xi-0.01, eta))/0.02
        dN_deta = (self.shape_functions(xi, eta+0.01) - self.shape_functions(xi, eta-0.01))/0.02
        return np.array(( np.array((dN_dxi, np.zeros_like(dN_dxi))).T.flatten(),
                          np.array((np.zeros_like(dN_deta), dN_deta)).T.flatten(),
                          np.array((dN_deta, dN_dxi)).T.flatten()
                          ))

    def _G(self, xi=0, eta=0):
        SS = self.shape_functions(xi, eta)
        dN_dxi = (self.shape_functions(xi + 0.01, eta) - SS)/0.01
        dN_deta = (self.shape_functions(xi, eta + 0.01) - SS)/0.01
        return np.array((dN_dxi, dN_deta))

    def displacements(self):
        return np.array([node.displacements for node in self.nodes])

    def displacements_(self, eta=0, xi=0):
        N = self.shape_functions(eta, xi)
        return np.array((np.array((N, np.zeros_like(N))).T.flatten(),
                         np.array((np.zeros_like(N), N)).T.flatten()
                         )) @ self.displacements().flatten()

    def strain(self, eta=0, xi=0):
        return self.B(eta,xi) @ self.displacements().flatten()

    def stress(self, eta=0, xi=0):
        return self.C @ self.strain(eta,xi)

    def stress_vonmises(self, eta=0, xi=0):
        sxx, syy, sxy = self.stress(eta, xi)
        return np.sqrt(sxx**2 - sxx*syy + syy**2 + 3*sxy**2)

    def Ex(self, newdim):
        dofs = np.array([node.dofs for node in self.nodes]).flatten()
        E = np.zeros((newdim, len(dofs)))
        for i, j in enumerate(dofs):
            E[j, i] = 1
        return E

    def __eq__(self, other):
        return np.allclose(self.r, other.r)

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

        super().__init__()

        self.r = r
        #self.n_integration_points = 2

    def shape_functions(self, xi=0, eta=0):
        return 1/4 * np.array([(1-xi)*(1-eta),
                               (1+xi)*(1-eta),
                               (1+xi)*(1+eta),
                               (1-xi)*(1+eta)])

    def G(self, xi=0, eta=0):
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
        super().__init__()

        self.r = r
        if self.r.shape == (4,2):
            self.r = np.array([*r,
                               (r[0]+r[1])/2,
                               (r[1]+r[2])/2,
                               (r[2]+r[3])/2,
                               (r[3]+r[0])/2])

    def shape_functions(self, xi=0, eta=0):
        # Nodes 1-4 are corner nodes and 5-8 are midside nodes
        return 1/4 * np.array([(1-xi)*(eta-1)*(xi+eta+1),
                               (1+xi)*(eta-1)*(eta-xi+1),
                               (1+xi)*(1+eta)*(xi+eta-1),
                               (xi-1)*(eta+1)*(xi-eta+1),
                               2*(1-eta)*(1-xi**2),
                               2*(1+xi)*(1-eta**2),
                               2*(1+eta)*(1-xi**2),
                               2*(1-xi)*(1-eta**2)])

    def G(self, xi=0, eta=0):
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
        super().__init__()

        self.plotidx = [0,1,2]
        self.r = r

    def _shape_functions(self, xi=0, eta=0):
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

    def shape_functions(self, xi=0, eta=0):
        return np.array([eta,
                         xi,
                         1-eta-xi])

class Tri6(FiniteElement):
    def __init__(self, r, *args, **kwargs):
        super().__init__()

        self.plotidx = [0,3,1,4,2,5]
        self.r = r
        if self.r.shape == (3,2):
            self.r = np.array([*r,
                               (r[0] + r[1]) / 2,
                               (r[1] + r[2]) / 2,
                               (r[0] + r[2]) / 2])

    def shape_functions(self, xi=0, eta=0):
        zeta = 1 - xi - eta
        return np.array([xi*(2*xi - 1),
                         eta*(2*eta - 1),
                         zeta*(2*zeta - 1),
                         4*xi*eta,
                         4*eta*zeta,
                         4*zeta*xi])

class Problem:
    def __init__(self):
        self.nodes = list()
        self.elements = list()

        self.constrained_dofs = np.array([], dtype=int)
        self.loads = np.array([])
        self.displacements = np.array([])
        self.nodal_coordinates = self.f_nodal_coordinates()
        self.nc_dict = dict()

    def create_element(self, r, cls=Quad4, E=2e5, nu=0.3, thickness=1):
            element = cls(r, E, nu, thickness)
            self.elements.append(element)
            element.number = self.elements.index(element)
            for point in element.r:
                node = self.create_node(point)
                node.elements.append(element)
                element.nodes.append(node)

    def create_node(self, r):
        r = np.asarray(r)
        node = Node(r)
        if tuple(r) not in self.nc_dict:
            self.nodes.append(node)
            self.nc_dict[tuple(r)] = len(self.nodes) - 1
            node.number = len(self.nodes) - 1
            node.dofs = 2*node.number + np.array([0, 1])
            return node
        else:
            #print('Node already exists at', r)
            return self.nodeobj_at(r)
            pass

    def mesh_grid(self, x, y, cls=Quad4):
        i = 0
        for x1, x2 in zip(x, x[1:]):
            for y1, y2 in zip(y, y[1:]):
                i += 1
                r = np.array([[x1, y1],
                              [x2, y1],
                              [x2, y2],
                              [x1, y2]])
                if cls.__name__[0:4] == 'Quad':
                    self.create_element(r, cls=cls)
                elif cls == Tri3 or cls == Tri6:
                    xm, ym = 1/2 * np.array([x1+x2, y1+y2])
                    for (xa,ya),(xb,yb) in zip(r, np.roll(r,1,axis=0)):
                        r = np.array([[xa, ya],
                                      [xb, yb],
                                      [xm, ym]])
                        self.create_element(r, cls=cls)

    def load_node(self, r, loads):
        self.nodeobj_at(r).loads = np.asarray(loads)
        self.loads = np.array([node.loads for node in self.nodes]).flatten()

    def pin(self, node):
        self.constrained_dofs = np.append(self.constrained_dofs, node.dofs)

    def stiffness_matrix(self, reduced=False):
        self.nodal_coordinates = self.f_nodal_coordinates()
        ndofs = 2 * len(self.nodes)
        K = np.zeros((ndofs, ndofs))
        for element in self.elements:
            E = element.Ex(ndofs)
            K = K + E @ element.stiffness_matrix() @ E.T
        if not reduced:
            return K
        elif reduced:
            K = np.delete(K, self.constrained_dofs, axis=0)
            K = np.delete(K, self.constrained_dofs, axis=1)
            return K

    def free_dofs(self):
        return np.delete(np.arange(2*len(self.nodes)), self.constrained_dofs)

    def solve(self):
        free = self.free_dofs()
        Kr = self.stiffness_matrix(reduced=True)

        self.displacements = np.zeros(2*len(self.nodes))
        self.displacements[free] = np.linalg.solve(Kr, self.loads[free])
        for node in self.nodes:
            node.displacements = self.displacements[node.dofs]

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

    def plot_prep(self, ncols=1):
        fig,axs = plt.subplots(ncols=ncols)
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



def R2(theta):
    s, c = np.sin(theta), np.cos(theta)
    return np.array([[c, -s], [s, c]])


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
        p.mesh_grid(x, y, cls=cls)
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
            p.mesh_grid(x, y, cls=cls)
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
        p.mesh_grid(x, y, cls=Quad8)

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


        p.create_element(r1, cls=cls)

        p.load_node(R@np.array([10,10]), R@np.array([10000,0]))
        #p.pin(p.nodeobj_at((0,1)))
        p.pin(p.nodeobj_at(R@np.array([0,10])))
        p.pin(p.nodeobj_at(R@np.array([0,1])))
        p.solve()
        elm = p.elements[0]

        #p.plot_functions(p.plot_model, (p.plot_displaced, p.plot_loads))
        p.animate(); plt.show()









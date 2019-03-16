import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
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
        self.C = E/(1-nu)**2 * np.array([   [1, nu, 0],
                                            [nu, 1, 0],
                                            [0, 0, (1-nu)/2]])

    def stiffness_matrix(self, n_integration_points = 2):
        """

        :param n_integration_points: Number of integration points PER DIRECTION. 2 or 3 supported
        :return:
        """

        if n_integration_points == 2:
            a = 1/np.sqrt(3)
            integration_points = np.array([[-a, -a], [ a, -a],
                                           [ a,  a], [-a,  a]])
            weights = np.array([1, 1, 1, 1])

        elif n_integration_points == 3:
            a = np.sqrt(3/5)
            integration_points = np.array([[-a, a],  [0, a],  [a, a],
                                           [-a, 0],  [0, 0],  [a, 0],
                                           [-a, -a], [0, -a], [a,-a]])
            weights = 1/81 * np.array([25, 40, 25,
                                       40, 64, 40,
                                       25, 40, 25])
            # weights = np.outer([1, 5/8, 1], [1, 5/8, 1]) * 25/81 .flatten()

        k = sum(self.B(*r).T @ self.C @ self.B(*r) * self.thickness * np.linalg.det(self.jacobian(*r)) * w
                for r,w in zip(integration_points, weights))
        return k

    def jacobian(self, xi=0, eta=0):
        return self.G(xi, eta) @ self.r

    def B(self, xi=0, eta=0):
        dN_dx, dN_dy = np.linalg.inv(self.jacobian(xi, eta)) @ self.G(xi, eta)
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

    def G(self, xi=0, eta=0):
        dN_dxi = (self.shape_functions(xi + 0.01, eta) - self.shape_functions(xi - 0.01, eta))/0.02
        dN_deta = (self.shape_functions(xi, eta + 0.01) - self.shape_functions(xi, eta - 0.01))/0.02
        return np.array((dN_dxi, dN_deta))

    def displacements(self):
        return np.array([node.displacements for node in self.nodes])

    def stress(self, eta=0, xi=0):
        return self.C @ self.B(eta,xi) @ self.displacements().flatten()

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

        self.plotidx = [0,1,2,3]
        self.r = r

    def shape_functions(self, xi=0, eta=0):
        return 1/4 * np.array([(1-xi)*(1-eta),
                               (1+xi)*(1-eta),
                               (1+xi)*(1+eta),
                               (1-xi)*(1+eta)])


class Quad8(FiniteElement):
    def __init__(self, r, *args, **kwargs):
        """
                :param r:  8x2 array of coordinates: [[x1,y1], [x2, y2], .. ]
                If a 4x2 array of coordinates is passed, will linearly interpolate midnodes between vertex nodes
                Nodes 5, 6, 7, 8 are midside nodes
                """
        super().__init__()

        self.plotidx = [0,4,1,5,2,6,3,7]
        if r.shape == (4,2):
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

class Problem:
    def __init__(self):
        self.nodes = list()
        self.elements = list()

        self.constrained_dofs = np.array([])
        self.loads = np.array([])
        self.displacements = np.array([])
        self.nodal_coordinates = self.f_nodal_coordinates()
        self.nc_dict = dict()

    def create_element(self, r, cls=Quad8, E=2e5, nu=0.3, thickness=1):
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
                self.create_element(r, cls=cls)

    def load_node(self, r, loads):
        self.nodeobj_at(r).loads = np.asarray(loads)
        self.loads = np.array([node.loads for node in self.nodes]).flatten()

    def pin(self, node):
        self.constrained_dofs = np.append(self.constrained_dofs, node.dofs)

    def stiffness_matrix(self, reduced=False, all_equal=True):
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

    def plot(self): 
        fig,ax = plt.subplots()

        for element in self.elements:
            r = element.r
            rect = patches.Rectangle(r[0], *(r[2]-r[0]), linewidth=1, facecolor='none')
            ax.add_patch(rect)
        plt.autoscale()
        plt.show()

    def plot_displaced(self, save=False, filename='fig.jpg'):
        fig,ax = plt.subplots()
        xmin, xmax = np.min(self.f_nodal_coordinates().T[0]), np.max(self.f_nodal_coordinates().T[0])+20
        ymin, ymax = np.min(self.f_nodal_coordinates().T[1]), np.max(self.f_nodal_coordinates().T[1])+100
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        elemental_stress = np.array([element.stress(0,0)[0] for element in self.elements])
        maxstress = np.max(elemental_stress)*1.01
        ax.text(0.1*xmax, 0.8*ymax, 'Max stress {}'.format(maxstress))

        for element in self.elements:
            r = element.r[element.plotidx] + element.displacements()[element.plotidx]
            polygon = patches.Polygon(r, linewidth=1, facecolor=(np.sqrt(np.abs(element.stress(0,0)[0]/maxstress)),
                                                                 0, 0))
                                                                 #1 - np.sqrt(element.stress_vonmises(0,0)/maxstress)))
            ax.add_patch(polygon)
        if save:
            plt.savefig(filename)

        #plt.show()


if __name__ == '__main__':

    for n in (1,2,3,4,5):
        start_time = time.time()
        r1 = np.array([[0,0],
                      [10,0],
                      [10,10],
                      [0,10]])


        p = Problem()
        #n = 4
        x = 10*np.linspace(0,10, 10*n+1)
        y = 10*np.linspace(0,2, 2*n+1)
        cls = Quad8
        p.mesh_grid(x, y, cls=cls)
        mid_time = time.time()
        p.load_node((100,0), (0,10000))
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
        e1 = p.nodeobj_at((80, 10)).elements[0]
        e2 = p.nodeobj_at((10, 20)).elements[0]
        e3 = p.nodeobj_at((10, 0)).elements[0]

        dt = end_time - start_time
        print('n = ', n)
        print('Mesh time', mid_time - start_time)
        print('Assembly and solution time', end_time - mid_time)
        print('Total time', dt)
        filename = '{}_{}_16-03-2019.png'.format(cls.__name__, n)
        p.plot_displaced(save=True, filename=filename)

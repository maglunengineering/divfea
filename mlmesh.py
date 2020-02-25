import itertools as it
from typing import List, Any

import numpy as np
import matplotlib.pyplot as plt

def R(theta):
    s,c = np.sin(theta), np.cos(theta)
    return np.array([[c, -s],[s, c]])


class Vertex:
    r : np.ndarray

    def __init__(self, r:np.ndarray):
        self.r = np.asarray(r)
        self.edges = list()
        self.is_apex = False

        self.edges = list()
        self.elements = list()

    def transform_by(self, func):
        self.r = func(self.r)
        return self

    def add_element(self, element):
        self.elements.append(element)

    def add_edge(self, edge):
        self.edges.append(edge)

    def move_by(self, dr):
        self.r += dr

    def move_to(self, r):
        self.r = r

    def get_neighbours(self):
        neighbours = []
        for element in self.elements:
            for vertex in element.vertices:
                if vertex != self and vertex not in neighbours:
                    neighbours.add(vertex)
        return neighbours

    def __sub__(self, other):
        return other.r - self.r

    def __repr__(self):
        return self.r.round(3).__repr__().replace('array', 'apex' if self.is_apex else 'vertex')

    def __hash__(self):
        return tuple(self.r).__hash__()

class Line2D:
    seeds: np.ndarray
    v1:Vertex
    v2:Vertex

    def __init__(self, v1, v2):
        if type(v1) is np.ndarray or type(v1) is tuple:
            self.v1 = Vertex(v1)
            self.v2 = Vertex(v2)
        else:
            self.v1 = v1
            self.v2 = v2

        self.v1.add_edge(self)
        self.v2.add_edge(self)

        self.seeds = None

    @property
    def r(self):
        return np.array((self.r1,self.r2))

    @property
    def r1(self):
        return self.v1.r

    @property
    def r2(self):
        return self.v2.r

    @property
    def num_seeds(self):
        if self.seeds is None:
            return 0
        else:
            return len(self.seeds)

    def tangent(self):
        return (self.r2 - self.r1) / np.linalg.norm(self.r2 - self.r1)

    def left_normal(self):
        """
        Normalized
        :return:
        """

        m = np.array([[0, -1], [1, 0]])
        n = m @ (self.r2 - self.r1)
        n = n / np.linalg.norm(n)
        return  n

    def length(self):
        return np.linalg.norm(self.r2 - self.r1)

    def split(self, num_total: int) -> np.ndarray:
        num_total += 2
        retval = np.linspace(self.r1, self.r2, num_total)
        self.seeds = retval[range(1,num_total-1)]
        return retval[range(1,num_total-1)]

    def intersect(self, other, in_domain=False):
        """

        :param other:
        :param in_domain: If True, only vertices in the domain of the lines are considered
        :return:
        """
        t1 = self.r2 - self.r1
        t2 = other.r2 - other.r1

        M = np.array((t1, t2)).T
        if np.linalg.matrix_rank(M) < 2: # Singular matrix: parallell lines: return None
            return None

        ab = np.linalg.solve(M, self.r1 - other.r1)

        if np.abs(ab[0]) == np.abs(ab[1]) and 0 <= np.abs(ab[0]) <= 1 or not in_domain:
            return self.r1 - ab[0] * t1
        else:
            return None

    def left_or_right(self, r):
        """
        :return: 1 (left), -1 (right), 0 (on line)
        """
        det = np.linalg.det(self.r - r)
        if det == 0:
            return 0
        if det > 0:
            return 1
        if det < 0:
            return -1

    def contains_point(self, r) -> bool:
        kkT = (r - self.r1) / (self.r2 - self.r1)
        if np.allclose(kkT[0], kkT) and 0 <= kkT[0] <= 1:
            return True
        return False

    def absolute_to_normalized(self, r) -> float:
        dr = self.r2 - self.r1
        kk = np.divide(r[r != 0] , dr[dr != 0])
        if np.allclose(kk[0], kk[~np.isnan(kk)]):
            return kk[0]

    def normalized_to_absolute(self, flt) -> np.ndarray:
        return self.r1 + flt * (self.r2 - self.r1)

    def element_size(self):
        return self.length() / self.num_seeds

    def plot(self, fig, ax):
        ax.plot(*self.r.T, color='C2')
        if self.seeds is not None:
            for seed in self.seeds:
                ax.plot(*seed, 'ko')

    def __eq__(self, other):
        return np.allclose(self.r, other.r)

    def __hash__(self):
        return (tuple(self.r1) + tuple(self.r2)).__hash__()

    def __repr__(self):
        return "Line between {} and {}".format(self.r1, self.r2)

    def __add__(self, other):
        return Line2D(self.r1 + other, self.r2 + other)

class LineChain:
    edges: List[Line2D]

    def __init__(self, list_of_vertices: List[Vertex]):
        if type(list_of_vertices[0]) is np.ndarray or type(list_of_vertices[0]) is tuple:
            self.vertices = [Vertex(r) for r in list_of_vertices]
        else:
            self.vertices = np.asarray(list_of_vertices)
        self.edges = list()
        self.edge_me()

    def edge_me(self):
        self.edges = list()
        for v1, v2 in zip(self.vertices, self.vertices[1:]):
            edge = Line2D(v1,v2)
            self.edges.append(edge)
            v1.add_edge(edge)
            v2.add_edge(edge)

    def interior_angles(self) -> np.ndarray:
        """
        Angles at corners (r1, r2, .., rn), same order
        """
        angles = []
        for i in range(len(self.vertices)):
            prev = self.vertices[i - 1].r
            this = self.vertices[i].r
            next = self.vertices[(i + 1) % len(self.vertices)].r
            t1 = (this - prev) / np.linalg.norm(this - prev)
            t2 = (next - this) / np.linalg.norm(next - this)
            angl = np.pi - np.arccos(t1 @ t2)
            angles.append(angl)
        return angles

    def total_length(self) -> float:
        return sum(e.length() for e in self.edges)

    def find_edge_starting_at(self, r:np.ndarray) -> Line2D:
        for edge in self.edges:
            if np.allclose(edge.v1.r, r):
                return edge
        return None

    def find_edge_ending_at(self, r:np.ndarray) -> Line2D:
        for edge in self.edges:
            if np.allclose(edge.v2.r, r):
                return edge
        return None

    def next_vertex(self, r:np.ndarray) -> Vertex:
        e = self.find_edge_starting_at(r)
        return e.v2

    def prev_vertex(self, r:np.ndarray) -> Vertex:
        e = self.find_edge_ending_at(r)
        return e.v1

    def next_edge(self, e) -> Line2D:
        return self.edges[self.edges.index(e) + 1]

    def prev_edge(self, e) -> Line2D:
        return self.edges[self.edges.index(e) - 1]

    def seed_all_by_number(self, num):
        for edge in self.edges:
            edge.seeds = edge.split(num)

    def seed_all_by_elm_size(self, size):
        for edge in self.edges:
            num = int(np.round(edge.length() / size))
            edge.split(num)

    def seed_idx_by_number(self, edge_idx, num):
        self.edges[edge_idx].split(num)

    def get_seeds_by_edge_idx(self, edge_idx) -> np.ndarray:
        return self.edges[edge_idx].seeds

    def get_seeds_by_edge(self, edge) -> np.ndarray:
        return edge.seeds

    def plot(self, fig, ax):
        for edge in self.edges:
            edge.plot(fig, ax)

    @staticmethod
    def from_edges(self):
        pass

    def __eq__(self, other):
        return np.allclose(self.vertices, other.points)

    def __iter__(self):
        yield from self.vertices

class Loop(LineChain):
    edges: List[Line2D]

    def normal(self):
        pass

    def edge_me(self):
        LineChain.edge_me(self)
        last_edge = Line2D(self.vertices[-1], self.vertices[0])
        self.edges.append(last_edge)
        self.vertices[-1].add_edge(last_edge)
        self.vertices[0].add_edge(last_edge)

    def point_curvature(self, r) -> int:
        """
        1 : Loop curves inward at this point (convex)
        0 : Edges are parallell
        -1 : Loop curves outward at this point (concave)
        """
        e1 = self.find_edge_ending_at(r)
        e2 = self.find_edge_starting_at(r)
        return e1.left_or_right(e2.r[1])

    def get_first_convex_vertex(self) -> Vertex:
        for vertex in self.vertices:
            if self.point_curvature(vertex.r) < 0:
                return vertex
        return None

    def split_from_concave(self) -> tuple:
        cvx = self.get_first_convex_vertex()
        e1 = self.find_edge_starting_at(cvx.r)
        e2 = self.find_edge_ending_at(cvx.r)

        shortest_dist = np.inf
        closest_intersect = None
        split_edge = None
        for i, edge in enumerate(self.edges):
            if edge == e1 or edge == e2:
                continue

            n = edge.left_normal()
            split_line = Line2D(cvx.r, cvx.r + n)
            intersection = edge.intersect(split_line)
            this_dist = np.linalg.norm(intersection - cvx.r)

            if this_dist < shortest_dist:
                shortest_dist = this_dist
                closest_intersect = Vertex(intersection)
                split_edge = edge

        # Now we have the shortest line
        left_vertices = [cvx, closest_intersect, split_edge.v2]
        next_vtx = self.next_vertex(split_edge.r2)

        while not np.allclose(next_vtx.r, cvx.r):
            left_vertices.append(next_vtx)
            next_vtx = self.next_vertex(next_vtx.r)

        right_vertices = [cvx, closest_intersect, split_edge.v1]
        prev_vtx = self.prev_vertex(split_edge.r2)

        while not np.allclose(prev_vtx.r, split_edge.r1):
            right_vertices.append(prev_vtx)
            prev_vtx = self.prev_vertex(prev_vtx.r)
        right_vertices.reverse()

        right_loop = Loop(right_vertices)
        left_loop = Loop(left_vertices)

        right_loop.edge_me()
        left_loop.edge_me()

        return (left_loop, right_loop)


class ElementaryMesh:
    def __init__(self):
        pass

class EQ53(ElementaryMesh):
    n1 = 5
    n2 = 3
    def __init__(self, r1, r2, r3, r4):
        self.r1 = r1
        self.r2 = r2
        self.r3 = r3
        self.r4 = r4
        pass

    @property
    def external_nodes(self):
        return np.array((self.r1, self.r2, self.r3, self.r4))

    @property
    def internal_nodes(self):
        return self.mapper_matrix() @ self.external_nodes

    def mapper_matrix(self):
        return np.array([[3/4, 1/4,   0,   0],
                         [1/2, 1/2,   0,   0],
                         [1/4, 3/4,   0,   0],
                         [  0, 1/2, 1/2,   0],
                         [  0,   0, 1/2, 1/2],
                         [1/2,   0,   0, 1/2],
                         [1/2,   0, 1/2,   0],
                         [1/3,   0, 2/3,   0],
                         [1/3,   0, 1/3,   0]])

class GWithShapeFunctions:

    @property
    def r(self) -> np.ndarray:
        pass

    def N(self, xi, eta) -> np.ndarray:
        pass

    def G(self, xi, eta) -> np.ndarray:
        pass

    def jacobian(self, xi, eta) -> np.ndarray:
        return self.G(xi, eta) @ self.r

    def as_loop(self) -> Loop:
        return Loop([r for r in self.r])

    def plot(self, fig, ax):
        rr = np.vstack((self.r, self.r[0]))
        ax.plot(*rr.T, color='black')
        for vertex in self.vertices:
            ax.plot(*vertex.r, 'go', markersize=2)

class Quadrilateral(GWithShapeFunctions):
    def __init__(self, v1, v2, v3, v4):
        self.v1 = Vertex(v1) if type(v1) is np.ndarray or type(v1) is tuple else v1
        self.v2 = Vertex(v2) if type(v2) is np.ndarray or type(v2) is tuple else v2
        self.v3 = Vertex(v3) if type(v3) is np.ndarray or type(v3) is tuple else v3
        self.v4 = Vertex(v4) if type(v4) is np.ndarray or type(v4) is tuple else v4
        self.vertices = (v1,v2,v3,v4)

        self.v1.add_element(self)
        self.v2.add_element(self)
        self.v3.add_element(self)
        self.v4.add_element(self)

    @property
    def r(self):
        return np.array([self.v1.r, self.v2.r, self.v3.r, self.v4.r])

    def N(self, xi, eta):
        return 1 / 4 * np.array([(1 - xi) * (1 - eta),
                                 (1 + xi) * (1 - eta),
                                 (1 + xi) * (1 + eta),
                                 (1 - xi) * (1 + eta)])

    def G(self, xi, eta):
        return 1/4 * np.array([[ -(1-eta), 1-eta,  1+eta, -(1+eta)],
                               [ -(1-xi), -(1+xi), 1+xi,   1-xi]])

    def diag_skew(self):
        l1 = np.linalg.norm(self.v1.r - self.v3.r)
        l2 = np.linalg.norm(self.v2.r - self.v4.r)
        return np.max((l1/l2, l2/l1))

    def aspect_ratio(self):
        r = self.r
        r_rolled = np.roll(r, 1, axis=0)
        side_lengths = np.linalg.norm(r_rolled - r, axis=1)
        return np.max(side_lengths) / np.min(side_lengths)

    def area(self):
        x,y = self.r.T
        return 1/2 * (x[:-1] @ y[1:] - x[1:] @ y[:-1])

    def centroid(self):
        x,y = self.r.T
        A = self.area()
        cx = 1/(6*A) * (x[:-1] + x[1:]) @ (x[:-1]*y[1:] - x[1:]*y[:-1])
        cy = 1/(6*A) * (y[:-1] + y[1:]) @ (x[:-1]*y[1:] - x[1:]*y[:-1])
        return np.array([cx,cy])

    def adjacent_elements(self, of_degree=1):
        adjacents = []
        for v in self.vertices:
            for e in v.elements:
                if e == self:
                    continue
                common_vertices = [v for v in e.vertices if v in self.vertices]
                if len(common_vertices) > of_degree:
                    adjacents.append(e)
        return adjacents

    def is_self_intersecting(self) -> bool:
        edges = []
        for v1,v2 in zip(self.vertices, self.vertices[1:]):
            edges.append(Line2D(v1, v2))
        edges.append(Line2D(self.vertices[-1], self.vertices[0]))
        for edge1 in edges:
            for edge2 in edges:
                if edge1 == edge2:
                    continue
                intersect = edge1.intersect(edge2, in_domain=True)
                if intersect is not None and not np.allclose(edge1.r1, intersect) and not np.allclose(edge1.r2, intersect):
                    return True
        return False


class NormalizedTriangle(GWithShapeFunctions):
    """
    r1: At origin
    Then counterclockwise
    """
    def __init__(self, v1, v2, v3):
        self.v1 = v1
        self.v2 = v2
        self.v3 = v3

    @property
    def r(self):
        return np.array([self.v1.r, self.v2.r, self.v3.r])

    def N(self, xi, eta):
        return np.array([1 - eta - xi,
                         xi,
                         eta])

    def G(self, xi, eta) -> np.ndarray:
        return np.array([[-1, 1, 0]
                         [-1, 0, 1]])

class StaticInterpolatorTriangle:
    def __init__(self, rr, rr_, origin=(0,0)):
        """

        :param rr: Original coordinates
        :param rr_: Transformed coordinates
        """

        self.origin = np.asarray(origin)
        self.rr = np.asarray(rr)
        self.rr_ = np.asarray(rr_)

        self.transformation = (self.rr_ - self.origin).T @ np.linalg.inv(self.rr - self.origin).T

    def transform(self, pt:np.ndarray):
        return self.transformation @ (pt - self.origin) + self.origin

    def detransform(self, pt:np.ndarray):
        return np.linalg.solve(self.transformation, pt - self.origin) + self.origin

    def contains_point(self, pt:np.ndarray):
        lines = [
            Line2D(self.rr[0], self.rr[1]),
            Line2D(self.rr[1], self.origin),
            Line2D(self.origin, self.rr[0])]
        sides = [line.left_or_right(pt) for line in lines]
        return all(side == sides[0] for side in sides)

class QuadMesher:
    elements: List[Quadrilateral]

    def __init__(self):
        self.elements = []
        self.loops = []

    @staticmethod
    def mesh_structured_quad(quad:Quadrilateral, num_nodes_x:int, num_nodes_y:int):
        elements = []
        xx = np.linspace(-1, 1, num_nodes_x)
        yy = np.linspace(-1, 1, num_nodes_y)

        for xi1,xi2 in zip(xx, xx[1:]):
            for eta1,eta2 in zip(yy, yy[1:]):
                rr = np.array([[xi1, eta1],[xi2,eta1],[xi2,eta2],[xi1,eta2]])
                #r1, r2, r3, r4 = quad.N(*rr) @ quad.r
                """
                The above is shorthand for 
                r1 = quad.N(xi1, eta1) @ quad.r1
                r2 = quad.N(xi2, eta1) @ quad.r2
                and so on
                """
                r1 = quad.N(xi1, eta1) @ quad.r
                r2 = quad.N(xi2, eta1) @ quad.r
                r3 = quad.N(xi2, eta2) @ quad.r
                r4 = quad.N(xi1, eta2) @ quad.r

                q = Quadrilateral(r1, r2, r3, r4)
                elements.append(q)

        return elements

    def mesh_padding(self, loop:Loop):
        """
        Loop must be pre-seeded. Seeds must be excluding endpoints
        :param loop:
        :return:
        """
        self.loops.append(loop)
        for _ in range(1): # Boilerplate: Will be while(criterion based on smallness of inner loop)
            inner = self.offset_loop(loop)
            self.loops.append(inner)
            self.mesh_between_loops(loop, inner)

    def mesh_between_loops(self, outer:Loop, inner:Loop):
        """
        Outer loop must be pre-seeded
        """
        if not inner.edges:
            inner.edge_me()

        # No tuck

        for i in range(len(outer.edges)):
            if i == 0:
                pass # Do something

            outer_edge = outer.edges[i]
            inner_edge = inner.edges[i]

            rel = outer_edge.length() / inner_edge.length()
            assert rel >= 1 # Concave loops not yet supported

            if 1 - rel > outer_edge.element_size() / outer_edge.length():
                tuck_last = True
                inner_edge.split(outer_edge.num_seeds - 1, False)
            else:
                inner_edge.split(outer_edge.num_seeds, False)
                tuck_last = False

            outer_pts = outer_edge.seeds
            inner_pts = inner_edge.seeds

            for p1, p2, p3, p4 in zip(outer_pts, inner_pts, inner_pts[1:], outer_pts[1:]):
                q = Quadrilateral(p1, p2, p3, p4)
                self.elements.append(q)

    def offset_loop(self, loop:Loop):
        """
        Loop must be pre-seeded. Seeds must be excluding endpoints
        :param loop:
        :return:
        """
        #if len(loop.edges) == 0:
        #    loop.edge_me()
        #    loop.seed_all_by_number(5)

        inner = []
        #inner = [edge + edge.left_normal() * edge.length() / edge.num_seeds for edge in loop.edges]
        for vtx in loop.vertices:
            e0 = loop.find_edge_ending_at(vtx.r)
            e1 = loop.find_edge_starting_at(vtx.r)

            e0plus = e0 + e0.left_normal() * e0.element_size()
            e1plus = e1 + e1.left_normal() * e1.element_size()

            corner_pt = e0plus.intersect(e1plus)
            inner.append(corner_pt)

        newloop = Loop(inner)

        return newloop

    def plot(self, fig, ax):
        for l in self.loops:
            l.plot(fig, ax)
        for q in self.elements:
            q.plot(fig, ax)

class __QuadMesher:
    def __init__(self, loop:Loop, fig=None, ax=None):
        self.fig, self.ax = fig, ax
        if fig is None or ax is None:
            self.fig,self.ax = plt.subplots()
        self.loop = loop # Must be edged

        self.splitlines = list()
        self.crosslines = list()
        self.quads = list()
        self.tris = list()

        self.edges = loop.edges

        # Pick denseliest seeded edge as starting edge
        lengths = [edge.length() for edge in self.edges]
        first_edge_idx = lengths.index(max(lengths))


        first_edge = self.edges[first_edge_idx]
        left_edge = self.edges[first_edge_idx - 1]
        last_edge = self.edges[first_edge_idx - 2]
        right_edge = self.edges[first_edge_idx - 3]

        n1 = len(first_edge.seeds)
        if first_edge.tangent() @ last_edge.tangent() < 0:
            last_edge.seeds = last_edge.seeds[::-1]

        if right_edge.tangent() @ left_edge.tangent() < 0:
            left_edge.seeds = left_edge.seeds[::-1]


        delta = n1 - len(last_edge.seeds)
        assert len(left_edge.seeds) == len(right_edge.seeds)
        assert delta >= 0
        assert delta < len(left_edge.seeds)

        # For 3-to-2 reduction, delta must be smaller than the number of side seeds
        # Map first seeds to last seeds
        self.mapper = dict()
        f = n1 / len(last_edge.seeds)
        assert f > 1 and f < 2 # For now

        j = np.nan
        for i in range(n1):
            if int(np.round(i / f)) == j:
                continue ## This will be an "orphaned" node on first_edge
            j = int(np.round(i/f))
            self.mapper[i] = j
            splitline = Line2D(first_edge.seeds[i], last_edge.seeds[j])
            self.splitlines.append(splitline)

        self.plot_splitlines()

        for i, right_pt, left_pt in zip(it.count(), right_edge.seeds, left_edge.seeds):
            crossline = Line2D(right_pt, left_pt)
            self.crosslines.append(crossline)
            break

        self.plot_crosslines()

    def map_points(self, points1, points2):
        n1 = len(points1)
        n2 = len(points2)

    def plot_splitlines(self):
        for line in self.splitlines:
            self.ax.plot(*line.r.T, color='C1')

    def plot_crosslines(self):
        for line in self.crosslines:
            self.ax.plot(*line.r.T, color='C2')

class QuadPadMesher:
    vertices: List[Vertex]

    def __init__(self, loop:Loop):
        self.loop = loop
        self.triangles = list()
        self.elements = list()
        self.element_size = None
        self.vertices = []

        if not loop.edges:
            loop.edge_me()

        self.calculate_mesh_size()

        self.smoothing_factor = 1/50

    def calculate_mesh_size(self):
        # Better be pre-seeded
        self.element_size = np.linalg.norm(self.loop.edges[0].seeds[1] - self.loop.edges[0].seeds[0])

    def seed_and_triangle_generator(self, edges, triangles):
        for edge, tri in zip(edges, triangles):
            f = tri.detransform
            edge.v1.is_apex = True
            yield edge.v1.transform_by(f)
            for seed in edge.seeds:
                yield Vertex(seed).transform_by(f)
        yield edges[-1].v2.transform_by(f) # Eventually, this will re-yield the first vertex

    def mesh(self):
        elm_size = self.element_size
        straightened = self.straighten_and_triangulate()
        seeds = list()

        for r in self.seed_and_triangle_generator(straightened.edges, self.triangles):
            seeds.append(r)

        self.vertices.extend(seeds)

        for _ in range(5):
            seeds = self.pave_with_auto_tuck(seeds)
            self.vertices.extend(seeds)

        self.smooth()

    def pave_with_edge_tuck(self, outer_seeds:List[Vertex]) -> List[Vertex]:
        i = 0
        passed_apex = False
        midpt = self.midpoint()
        inner_seeds = []
        while i < (len(outer_seeds) - 1):
            v1 = outer_seeds[i]
            v3 = outer_seeds[i + 1]
            v2 = v2 if passed_apex \
                else v4 if i != 0 \
                else Vertex(v1.r + self.element_size * (midpt - v1.r) / np.linalg.norm(midpt - v1.r))
            v4 = Vertex(v3.r + self.element_size * (midpt - v3.r) / np.linalg.norm(midpt - v3.r))

            if not passed_apex:
                inner_seeds.append(v2)

            passed_apex = False
            if v3.is_apex:  # Corner tuck! New order will be v1, v2 (inner) v4 (reassigned, past the corner) and v3 (apex)
                del v4
                passed_apex = True
                v4 = outer_seeds[i + 2]  # Past the corner
                v2.is_apex = True  # Have v2 be an apex for the next round
                i += 1  # Increment i by one more

            quad = Quadrilateral(v1, v2, v4, v3)

            self.elements.append(quad)
            i += 1
        inner_seeds.append(v4)

        return inner_seeds

    def pave_without_tuck(self, outer_seeds:List[Vertex]) -> List[Vertex]:
        i = 0
        quads = []
        midpt = self.midpoint()
        inner_seeds = []
        while i < (len(outer_seeds) - 1):
            v1 = outer_seeds[i]
            v3 = outer_seeds[i + 1]
            v2 = v4 if i != 0 \
                else Vertex(v1.r + self.element_size * (midpt - v1.r) / np.linalg.norm(midpt - v1.r))
            v4 = Vertex(v3.r + self.element_size * (midpt - v3.r) / np.linalg.norm(midpt - v3.r))

            inner_seeds.append(v2)
            quad = Quadrilateral(v1, v2, v4, v3)
            quads.append(quad)
            i += 1
        inner_seeds.append(v4)

        self.elements.extend(quads)
        return inner_seeds

    def pave_with_auto_tuck(self, outer_seeds:List[Vertex]):
        i = 0
        quads = []
        midpt = self.midpoint()
        inner_seeds = []
        seed_lengths = [np.linalg.norm(v2 - v1) for v1,v2 in zip(outer_seeds[:-1], outer_seeds[1:])]
        total_length = sum(seed_lengths)
        n_seeds = len(outer_seeds)
        n_elements = int(np.round(total_length / self.element_size))
        n_tucks = n_seeds - n_elements - 1

        min_spacing = 5
        #i_with_tuck = np.round( np.linspace(0, n_seeds - 2, n_tucks + 2)[1:-1])
        indices = np.asarray(seed_lengths).argsort()[::1]
        i_with_tuck = [indices[0]]
        for idx in indices:
            if all([np.abs(i - idx) > min_spacing for i in i_with_tuck]):
                i_with_tuck.append(idx)
            if len(i_with_tuck) >= n_tucks:
                break

        while i < n_seeds - 1:
            v1 = outer_seeds[i]
            v3 = outer_seeds[i + 1]

            v2 = v4 if i != 0 \
                else Vertex(v1.r + self.element_size * (midpt - v1.r) / np.linalg.norm(midpt - v1.r))
            v4 = Vertex(v3.r + self.element_size * (midpt - v3.r) / np.linalg.norm(midpt - v3.r))

            passed_tuck = False
            if i in i_with_tuck and i < n_seeds - 2:
                passed_tuck = True
                v4 = v2
                v1 = outer_seeds[i]
                v2 = outer_seeds[i+1]
                v3 = outer_seeds[i+2]
                v4.move_to(v2.r + self.element_size * (midpt - v2.r) / np.linalg.norm(midpt - v2.r))

                quad = Quadrilateral(v1, v2, v3, v4)
                i += 1

                #if i < n_seeds - 3 and Quadrilateral(v1, v2, v3, outer_seeds[i+3]).diag_skew() < 2.5:
                #    trial_quad = Quadrilateral(v1, v2, v3, outer_seeds[i+3])
                #    if not trial_quad.is_self_intersecting():
                #        quad = trial_quad
                #        i += 1

            else:
                quad = Quadrilateral(v1, v2, v4, v3)
                inner_seeds.append(v2)

            quads.append(quad)
            i += 1

        inner_seeds.append(v4)
        self.elements.extend(quads)
        #self.smooth(inner_seeds)
        return inner_seeds

    def find_and_create_tucks(self, element_subset):
        for i in range(1, len(element_subset)-1):
            element = element_subset[i]
            if not element.aspect_ratio() > 2:
                continue

            prev = element_subset[i-1]
            next = element_subset[i+1]

            v1 = element.v2
            v2 = element.v3
            adj = element.adjacent_elements(of_degree=2)
            adj.remove(prev)
            adj.remove(next)
            rear = adj[0]
        pass

    def smooth(self, subset=[], factor=1/50):
        if not subset:
            subset = self.vertices # Smooth the entire mesh
        for vertex in subset:
            skew_0 = np.array([e.diag_skew() for e in vertex.elements])
            aspect_0 = np.array([e.aspect_ratio() for e in vertex.elements])

            vertex.move_by(np.array([1e-5, 0])) # dx
            skew_x = np.array([e.diag_skew() for e in vertex.elements]) - skew_0
            aspect_x = np.array([e.aspect_ratio() for e in vertex.elements]) - aspect_0

            vertex.move_by(np.array([-1e-5, 1e-5])) # dy
            skew_y = np.array([e.diag_skew() for e in vertex.elements]) - skew_0
            aspect_y = np.array([e.aspect_ratio() for e in vertex.elements]) - aspect_0
            vertex.move_by(np.array([0, -1e-5])) # 0

            D = np.array([skew_x, skew_y]).T
            ds = -skew_0 - 1

            dr = np.linalg.solve(D.T @ D, D.T) @ ds
            dr *= self.smoothing_factor * self.element_size / np.max(dr) # Move at maximum <factor> of the element size
            vertex.move_by(dr)


    def straighten_and_triangulate(self) -> LineChain:
        loop = self.loop
        straight_verts = list()
        start_vtx = loop.vertices[0]
        straight_verts.append(start_vtx)
        for edge in loop.edges:
            l = edge.length()
            end_vtx = Vertex(start_vtx.r + np.array((l, 0)))

            rr = edge.r
            rr_ = np.array((start_vtx.r, end_vtx.r))
            self.triangles.append(StaticInterpolatorTriangle(rr, rr_, self.midpoint()))

            start_vtx = end_vtx
            straight_verts.append(end_vtx)

        straight = LineChain(straight_verts)
        straight.edge_me()
        for straightedge, loopedge in zip(straight.edges, loop.edges):
            n = loopedge.num_seeds
            straightedge.split(n)
        return straight

    def midpoint(self):
        return np.average([v.r for v in self.loop.vertices], axis=0)

    def plot(self, fig, ax):
        for edge in self.loop.edges:
            edge.plot(fig, ax)
        for q in self.elements:
            q.plot(fig, ax)

if __name__ == "__main__":

    def a():
        fig, ax = plt.subplots()
        pts5 = np.array([[0, 0], [1, 1], [0, 2], [-1, 1.5], [-1, 0.5]])

        loop1 = Loop(pts5)
        loop1.edge_me()
        loop1.seed_all_by_number(7, False)
        qm = QuadMesher()
        qm.mesh_padding(loop1)

        qm.plot(fig, ax)

    def b():
        global qpm
        fig,ax = plt.subplots()

        ax.set_aspect('equal')

        pts = np.array([[0, 0], [1, 0.9], [0, 2], [-1, 1.5], [-1, 0.5]])
        loop = Loop(pts)

        loop.seed_all_by_elm_size(0.15)
        qpm = QuadPadMesher(loop)
        qpm.mesh()
        loop.plot(fig, ax)
        qpm.plot(fig, ax)

        plt.show()

    def c():
        global qpm
        fig,axes = plt.subplots(nrows=2, ncols=2)
        axes = axes.flatten()
        for ax in axes:
            ax.set_aspect('equal')

        pts = np.array([[0, 0], [1, 0.9], [0, 2], [-1, 1.5], [-1, 0.5]])
        loop = Loop(pts)

        for i,f in enumerate((0, )):
            loop.seed_all_by_elm_size(0.15)
            qpm = QuadPadMesher(loop)
            qpm.smoothing_factor = f
            qpm.mesh()
            loop.plot(fig, axes[i])
            qpm.plot(fig, axes[i])

        plt.show()

    b()




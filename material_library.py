import numpy as np

steelProperties = {'E1':210000, 'v12':0.3}

class Material:
    def __init__(self, *args, **kwargs):
        pass

    def StiffnessMatrix3D(self):
        pass

    def StiffnessMatrixPlaneStress(self):
        pass

    def StiffnessMatrixPlaneStrain(self):
        pass

    def ThermalExpansionCoeffs(self):
        """
        [alpha_1, alpha_2, alpha_3]
        :return:
        """
        pass

class MaterialIsotropic(Material):
    def __init__(self, properties):
        self.E = properties['E1']
        self.nu = properties['v12']
        self.alpha = properties['a1']

    def StiffnessMatrix3D(self):
        E = self.E
        nu = self.nu
        return E / ((1+nu) * (1-2*nu)) * np.array([ [1 - nu, nu, nu, 0, 0, 0],
                                                    [nu, 1 - nu, nu, 0, 0, 0],
                                                    [nu, nu, 1 - nu, 0, 0, 0],
                                                    [0, 0, 0,        0.5 - nu, 0, 0],
                                                    [0, 0, 0,        0, 0.5 - nu, 0],
                                                    [0, 0, 0,        0, 0, 0.5 - nu]])

    def StiffnessMatrixPlaneStress(self):
        E = self.E
        nu = self.nu
        return E / (1-nu**2) * np.array([[1,     nu,     0       ],
                                         [nu,    1,      0       ],
                                         [0,     0,   (1-nu)/2   ]])

    def StiffnessMatrixPlaneStrain(self):
        E = self.E
        nu = self.nu
        return E / ((1+nu)*(1-2*nu)) * np.array([[1-nu,     nu,     0],
                                                 [nu,       1-nu,   0],
                                                 [0,        0,      1-2*nu]])

    def ThermalExpansionCoeffs(self):
        return self.alpha * np.ones(3)

class MaterialTransverselyIsotropic(Material):
    def __init__(self):
        pass

class MaterialOrthotropic(Material):
    def __init__(self):
        pass

class MaterialLayered:
    def __init__(self, materials, orientations):
        """
        Layered material

        :param materials: List of materials from bottom to top
        :param orientations: List of orientations from bottom to top
        """
        assert len(materials) == len(orientations)

    def StressTransformation2D(self, angle):
        c,s = np.cos(angle), np.sin(angle)
        return np.array([[ c*c ,  s*s ,   2*c*s],
                         [ s*s ,  c*c ,  -2*c*s],
                         [-c*s,   c*s , c*c-s*s]], float)

    def StrainTransformation2D(self, angle):
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[   c*c,   s*s,     c*s ],
                         [   s*s,   c*c,    -c*s ],
                         [-2*c*s, 2*c*s, c*c-s*s ]], float)

    def StressTransformation3D(self, angle):
        c, s = np.cos(angle), np.sin(angle)
        return    np.array([[ c*c ,  s*s ,  0.0 , 0.0 , 0.0  ,  2*c*s ],
                            [ s*s ,  c*c ,  0.0 , 0.0 , 0.0  , -2*c*s ],
                            [ 0.0 ,  0.0 ,  1.0 , 0.0 , 0.0  ,    0.0 ],
                            [ 0.0 ,  0.0 ,  0.0 ,   c ,  -s  ,    0.0 ],
                            [ 0.0 ,  0.0 ,  0.0 ,   s ,   c  ,    0.0 ],
                            [-c*s ,  c*s ,  0.0 , 0.0 , 0.0  ,c*c-s*s ]],
                            float)

    def StrainTransformation3D(self, angle):
        c, s = np.cos(angle), np.sin(angle)
        return    np.array([[ c*c ,  s*s ,  0.0 , 0.0 , 0.0  ,    c*s ],
                            [ s*s ,  c*c ,  0.0 , 0.0 , 0.0  ,   -c*s ],
                            [ 0.0 ,  0.0 ,  1.0 , 0.0 , 0.0  ,    0.0 ],
                            [ 0.0 ,  0.0 ,  0.0 ,   c ,  -s  ,    0.0 ],
                            [ 0.0 ,  0.0 ,  0.0 ,   s ,   c  ,    0.0 ],
                            [-2*c*s, 2*c*s, 0.0 , 0.0 , 0.0  ,c*c-s*s ]],
                            float)

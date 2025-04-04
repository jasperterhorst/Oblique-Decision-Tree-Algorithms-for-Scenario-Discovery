import numpy as np
from scipy.spatial import HalfspaceIntersection
from scipy.optimize import linprog


def sort_polygon(points):
    center = np.mean(points, axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    return points[np.argsort(angles)]


class Region:
    def __init__(self, A=None, b=None):
        self.A = np.array(A) if A is not None else np.empty((0, 2))
        self.b = np.array(b) if b is not None else np.empty((0,))

    def add_constraint(self, a, b_val):
        self.A = np.vstack([self.A, a])
        self.b = np.append(self.b, b_val)

    def split(self, w, bias):
        region_left = Region(self.A.copy(), self.b.copy())
        region_right = Region(self.A.copy(), self.b.copy())
        region_left.add_constraint(-w, -bias)
        region_right.add_constraint(w, -bias)
        return region_left, region_right

    def to_polygon(self):
        print("\n🔍 Calling to_polygon()")
        print(f"   A = \n{self.A}")
        print(f"   b = {self.b}")

        try:
            lp_result = linprog(c=[0, 0], A_ub=self.A, b_ub=self.b)
            if not lp_result.success:
                print("   ❌ linprog: Infeasible region")
                return []
            feasible_point = lp_result.x
            center = np.array([0.5, 0.5])
            feasible_point = 0.99 * feasible_point + 0.01 * center
            print(f"   ✅ Feasible point found: {feasible_point}")
        except Exception as e:
            print(f"   ❌ linprog failed: {e}")
            return []

        try:
            hs = HalfspaceIntersection(np.hstack([self.A, -self.b[:, np.newaxis]]), feasible_point)
            print(f"   ✅ Polygon computed with {len(hs.intersections)} vertices")
            polygon = hs.intersections
            polygon = sort_polygon(polygon)
            return polygon
        except Exception as e:
            print(f"   ❌ HalfspaceIntersection failed: {e}")
            return []
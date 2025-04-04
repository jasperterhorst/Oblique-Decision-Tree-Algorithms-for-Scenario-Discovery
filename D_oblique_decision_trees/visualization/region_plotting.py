import numpy as np
import matplotlib.pyplot as plt
from D_oblique_decision_trees.core.tree import DecisionNode
from D_oblique_decision_trees.core.geometry import Region


def plot_oblique_tree(tree, X, y, xlim=(0, 1), ylim=(0, 1)):
    print("\n📦 Starting region-aware decision tree debug...")
    print(f"✅ Data shape: X={X.shape}, y={y.shape}")
    print(f"✅ Tree root: {tree.root}\n")

    xmin, xmax = xlim
    ymin, ymax = ylim

    root_region = Region(
        A=[[1, 0], [-1, 0], [0, 1], [0, -1]],
        b=[xmax, -xmin, ymax, -ymin]
    )

    def clip_line_to_polygon(w, b, polygon):
        if polygon is None or len(polygon) < 3:
            print("⚠️ Polygon too small or None")
            return None
        points = []
        print(f"   ➤ Clipping: w={w}, b={b}, polygon points={len(polygon)}")
        for i in range(len(polygon)):
            p1 = polygon[i]
            p2 = polygon[(i + 1) % len(polygon)]
            d1 = np.dot(w, p1) + b
            d2 = np.dot(w, p2) + b
            print(f"     • Edge {i}: d1={d1:.3f}, d2={d2:.3f}")
            if d1 * d2 < 0:
                t = d1 / (d1 - d2)
                intersection = p1 + t * (p2 - p1)
                points.append(intersection)
        if len(points) == 2:
            return points
        return None

    def recurse(node, region, depth=0):
        prefix = "    " * depth
        print(f"{prefix}🧠 Visiting node {node.node_id}, depth {node.depth}")

        if isinstance(node, DecisionNode):
            print(f"{prefix}🔹 DecisionNode: w={node.weights}, b={node.bias}")
            print(f"{prefix}🔹 Region has {len(region.A)} constraints")

            polygon = region.to_polygon()
            if polygon is not None and len(polygon) >= 3:
                print(f"{prefix}✅ Polygon with {len(polygon)} points")
            else:
                print(f"{prefix}❌ Cannot build polygon")

            fig, ax = plt.subplots(figsize=(5, 5))
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_aspect("equal")
            ax.set_title(f"Node {node.node_id} at depth {depth}")

            if polygon is not None and len(polygon) >= 3:
                poly_closed = np.vstack([polygon, polygon[0]])
                ax.fill(poly_closed[:, 0], poly_closed[:, 1], alpha=0.4, facecolor='lightblue', edgecolor='blue', linewidth=1.5, label='Region')

            segment = clip_line_to_polygon(node.weights, node.bias, polygon)
            if segment:
                p0, p1 = segment
                ax.plot([p0[0], p1[0]], [p0[1], p1[1]], 'r--', linewidth=2, label='Split line')
                print(f"{prefix}✅ Found clipped decision segment")
            else:
                print(f"{prefix}❌ No visible segment in region")

            ax.text(0.01, 0.99, f"ID {node.node_id}\nDepth {depth}", ha='left', va='top', transform=ax.transAxes)
            ax.legend()
            plt.tight_layout()
            plt.show()

            left_region, right_region = region.split(node.weights, node.bias)
            if len(node.children) > 0:
                recurse(node.children[0], left_region, depth + 1)
            if len(node.children) > 1:
                recurse(node.children[1], right_region, depth + 1)
        else:
            print(f"{prefix}🌿 LeafNode: prediction={getattr(node, 'prediction', None)}")

    recurse(tree.root, root_region)

    # def recurse(node, region, depth=0):
    #     prefix = "    " * depth
    #     print(f"{prefix}🧠 Visiting node {node.node_id}, depth {node.depth}")
    #
    #     if isinstance(node, DecisionNode):
    #         print(f"{prefix}🔹 DecisionNode: w={node.weights}, b={node.bias}")
    #         print(f"{prefix}🔹 Region has {len(region.A)} constraints")
    #
    #         polygon = region.to_polygon()
    #         if polygon is not None and len(polygon) >= 3:
    #             print(f"{prefix}✅ Polygon with {len(polygon)} points")
    #         else:
    #             print(f"{prefix}❌ Cannot build polygon")
    #
    #         segment = clip_line_to_polygon(node.weights, node.bias, polygon)
    #         if segment:
    #             print(f"{prefix}✅ Found clipped decision segment")
    #         else:
    #             print(f"{prefix}❌ No visible segment in region")
    #
    #         left_region, right_region = region.split(node.weights, node.bias)
    #         if len(node.children) > 0:
    #             recurse(node.children[0], left_region, depth + 1)
    #         if len(node.children) > 1:
    #             recurse(node.children[1], right_region, depth + 1)
    #     else:
    #         print(f"{prefix}🌿 LeafNode: prediction={getattr(node, 'prediction', None)}")

    def recurse(node, region, depth=0):
        prefix = "    " * depth
        print(f"{prefix}🧠 Visiting node {node.node_id}, depth {node.depth}")

        if isinstance(node, DecisionNode):
            print(f"{prefix}🔹 DecisionNode: w={node.weights}, b={node.bias}")
            print(f"{prefix}🔹 Region has {len(region.A)} constraints")

            polygon = region.to_polygon()
            if polygon is not None and len(polygon) >= 3:
                print(f"{prefix}✅ Polygon with {len(polygon)} points")
            else:
                print(f"{prefix}❌ Cannot build polygon")

            # 🖼️ Begin plot
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect("equal")
            ax.set_title(f"Node {node.node_id} at depth {depth}")

            # Draw polygon region
            if polygon is not None and len(polygon) >= 3:
                poly_closed = np.vstack([polygon, polygon[0]])
                ax.fill(poly_closed[:, 0], poly_closed[:, 1], alpha=0.4, facecolor='lightblue', edgecolor='blue',
                        linewidth=1.5, label='Region')

            # Draw split line (if visible)
            segment = clip_line_to_polygon(node.weights, node.bias, polygon)
            if segment:
                p0, p1 = segment
                ax.plot([p0[0], p1[0]], [p0[1], p1[1]], 'r--', linewidth=2, label='Split line')
                print(f"{prefix}✅ Found clipped decision segment")
            else:
                print(f"{prefix}❌ No visible segment in region")

            # Draw node label
            ax.text(0.01, 0.99, f"ID {node.node_id}\nDepth {depth}", ha='left', va='top', transform=ax.transAxes)

            ax.legend()
            plt.tight_layout()
            plt.show()

            # Split and recurse
            left_region, right_region = region.split(node.weights, node.bias)
            if len(node.children) > 0:
                recurse(node.children[0], left_region, depth + 1)
            if len(node.children) > 1:
                recurse(node.children[1], right_region, depth + 1)

        else:
            print(f"{prefix}🌿 LeafNode: prediction={getattr(node, 'prediction', None)}")

    recurse(tree.root, root_region)

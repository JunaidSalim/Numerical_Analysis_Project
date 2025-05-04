import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
from typing import List, Tuple

def evaluateBezier(t: float, 
                   p0: Tuple[float, float], 
                   p1: Tuple[float, float],
                   guide0: Tuple[float, float],
                   guide1: Tuple[float, float]) -> Tuple[float, float]:
    """
    Evaluate Bézier curve at parameter t using equations (3.25) and (3.26) from the book.
    
    Args:
        t: Parameter value in [0,1]
        p0: First endpoint (x₀, y₀)
        p1: Second endpoint (x₁, y₁)
        guide0: Guide point for p0 (x₀ + α₀, y₀ + β₀)
        guide1: Guide point for p1 (x₁ - α₁, y₁ - β₁)
    
    Returns:
        Point (x,y) on the curve at parameter t
    """
    x0, y0 = p0
    x1, y1 = p1
    
    α0 = guide0[0] - x0
    β0 = guide0[1] - y0
    α1 = x1 - guide1[0]
    β1 = y1 - guide1[1]
    
    x = (2*(x0 - x1) + 3*(α0 + α1))*t**3 + \
        (3*(x1 - x0) - 3*(α1 + 2*α0))*t**2 + \
        3*α0*t + x0
    
    y = (2*(y0 - y1) + 3*(β0 + β1))*t**3 + \
        (3*(y1 - y0) - 3*(β1 + 2*β0))*t**2 + \
        3*β0*t + y0
    
    return x, y

def plotBezierCurve(p0: Tuple[float, float],
                     p1: Tuple[float, float],
                     guide0: Tuple[float, float],
                     guide1: Tuple[float, float],
                     num_points: int = 20) -> None:
    """
    Plot a Bézier curve with its control points and guide lines.
    """
    print("\nBézier Curve Parameters:")
    params_df = pd.DataFrame({
        'Point Type': ['Endpoint 1', 'Endpoint 2', 'Guide Point 1', 'Guide Point 2'],
        'X': [p0[0], p1[0], guide0[0], guide1[0]],
        'Y': [p0[1], p1[1], guide0[1], guide1[1]]
    })
    print(params_df.to_string(index=False))
    print("\n")
    
    t_values = np.linspace(0, 1, num_points)
    curve_points = [evaluateBezier(t, p0, p1, guide0, guide1) for t in t_values]
    x_values, y_values = zip(*curve_points)
    
    points_data = {
        't': t_values,
        'x(t)': x_values,
        'y(t)': y_values
    }
    points_df = pd.DataFrame(points_data)
    print("Points on the Bézier Curve:")
    print(points_df.to_string(index=False, float_format=lambda x: '{:.4f}'.format(x)))
    print("\n")
    
    plt.figure(figsize=(10, 8))
    plt.plot(x_values, y_values, 'b-', label='Bézier Curve')
    
    plt.plot([p0[0], guide0[0]], [p0[1], guide0[1]], 'r--', alpha=0.5)
    plt.plot([p1[0], guide1[0]], [p1[1], guide1[1]], 'r--', alpha=0.5)
    plt.plot([p0[0]], [p0[1]], 'ro', label='Endpoints')
    plt.plot([p1[0]], [p1[1]], 'ro')
    plt.plot([guide0[0]], [guide0[1]], 'go', label='Guide Points')
    plt.plot([guide1[0]], [guide1[1]], 'go')
    
    plt.axis('equal')
    plt.grid(True)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Cubic Bézier Curve (Book Implementation)')
    plt.legend()

def main():
    print("Starting Bézier curve calculation...")
    
    p0 = (0, 1)
    p1 = (3, 1)
    guide0 = (1, 3)
    guide1 = (2, -1)
    
    plotBezierCurve(p0, p1, guide0, guide1)
    print("Displaying plot... (Close the plot window to end the program)")
    plt.show()
    print("Program completed successfully")

if __name__ == "__main__":
    main()
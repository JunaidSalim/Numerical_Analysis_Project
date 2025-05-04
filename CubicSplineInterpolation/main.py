import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple, List

def originalFunction(x: float) -> float:
    """Original function f(x) = 1/(1 + x²) (Runge function)"""
    return 1/(1 + x**2)

def computeSplineCoefficients(
    xPoints: np.ndarray,
    yPoints: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the coefficients for cubic spline interpolation with natural boundary conditions.
    
    Args:
        xPoints: Array of x coordinates
        yPoints: Array of y coordinates
        
    Returns:
        Tuple of arrays (a, b, c, d) containing the coefficients for each spline segment
        where each segment is: a + b(x-x_i) + c(x-x_i)² + d(x-x_i)³
    """
    n = len(xPoints) - 1
    h = np.diff(xPoints)
    
    A = np.zeros((n+1, n+1))
    b = np.zeros(n+1)
    
    A[0, 0] = 1
    A[n, n] = 1
    
    for i in range(1, n):
        A[i, i-1] = h[i-1]
        A[i, i] = 2*(h[i-1] + h[i])
        A[i, i+1] = h[i]
        
        b[i] = 3*(yPoints[i+1] - yPoints[i])/h[i] - 3*(yPoints[i] - yPoints[i-1])/h[i-1]
    
    c = np.linalg.solve(A, b)
    
    a = yPoints[:-1]
    b = np.array([(yPoints[i+1] - yPoints[i])/h[i] - h[i]*c[i]/3 - h[i]*c[i+1]/6 
                  for i in range(n)])
    d = np.array([(c[i+1] - c[i])/(3*h[i]) for i in range(n)])
    
    return a, b, c[:-1], d

def evaluateSpline(
    x: float,
    xPoints: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray
) -> float:
    """
    Evaluate the cubic spline at point x.
    
    Args:
        x: Point at which to evaluate
        xPoints: Array of x coordinates
        a, b, c, d: Spline coefficients
        
    Returns:
        Value of spline at x
    """
    i = np.searchsorted(xPoints, x) - 1
    i = np.clip(i, 0, len(xPoints)-2)
    
    dx = x - xPoints[i]
    
    return a[i] + b[i]*dx + c[i]*dx**2 + d[i]*dx**3

def evaluatePoints(
    xPoints: np.ndarray,
    yPoints: np.ndarray,
    testPoints: List[float]
) -> List[dict]:
    """
    Evaluate spline interpolation at test points and collect results.
    
    Args:
        xPoints: Data points x coordinates
        yPoints: Data points y coordinates
        testPoints: Points at which to evaluate interpolation
        
    Returns:
        List of dictionaries containing evaluation data
    """
    a, b, c, d = computeSplineCoefficients(xPoints, yPoints)
    evaluationData = []
    
    for x in testPoints:
        yInterp = evaluateSpline(x, xPoints, a, b, c, d)
        yTrue = originalFunction(x)
        error = abs(yTrue - yInterp)
        evaluationData.append({
            'x': x,
            'yInterp': yInterp,
            'yTrue': yTrue,
            'error': error
        })
        
    return evaluationData

def plotResults(
    xPoints: np.ndarray,
    yPoints: np.ndarray,
    evaluationData: List[dict]
) -> None:
    """
    Plot the interpolation results
    """
    a, b, c, d = computeSplineCoefficients(xPoints, yPoints)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    xSmooth = np.linspace(min(xPoints), max(xPoints), 1000)
    ySmooth = [evaluateSpline(x, xPoints, a, b, c, d) for x in xSmooth]
    
    plt.plot(xSmooth, ySmooth, 'b-', label='Cubic Spline')
    plt.plot(xPoints, yPoints, 'ro', label='Data Points')
    plt.plot(xPoints, yPoints, 'gs', markersize=4, label='Knot Points')
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Cubic Spline Interpolation')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    ySmoothTrue = [originalFunction(x) for x in xSmooth]
    
    plt.plot(xSmooth, ySmoothTrue, 'g-', label='Original Function')
    plt.plot(xSmooth, ySmooth, 'b--', label='Cubic Spline')
    plt.plot(xPoints, yPoints, 'ro', label='Data Points')
    
    testX = [data['x'] for data in evaluationData]
    testYInterp = [data['yInterp'] for data in evaluationData]
    testYTrue = [data['yTrue'] for data in evaluationData]
    
    plt.plot(testX, testYInterp, 'c^', label='Interpolated Test Points')
    plt.plot(testX, testYTrue, 'm^', label='True Test Points')
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Comparison with Original Function')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def printResults(
    xPoints: np.ndarray,
    yPoints: np.ndarray,
    evaluationData: List[dict]
) -> None:
    """
    Print interpolation results in tabular format
    """
    print("\nCubic Spline Interpolation Results:")
    print("Original Data Points (Knots):")
    print("| Index |     x     |     y     |")
    print("|-------|-----------|-----------|")
    
    for i, (x, y) in enumerate(zip(xPoints, yPoints)):
        print(f"| {i:5d} | {x:9.4f} | {y:9.4f} |")
        
    print("\nSpline Evaluation at Test Points:")
    print("|    x     | Interpolated |  Function   |   Error    |")
    print("|----------|--------------|-------------|------------|")
    
    for data in evaluationData:
        print(f"| {data['x']:8.4f} | {data['yInterp']:10.4f} | {data['yTrue']:9.4f} | {data['error']:8.2e} |")

def main():
    xPoints = np.linspace(-5, 5, 15)
    yPoints = np.array([originalFunction(x) for x in xPoints])
    
    testPoints = np.array([-4.2, -3.1, -1.8, -0.7, 0.8, 2.3, 3.5, 4.4])
    
    evaluationData = evaluatePoints(xPoints, yPoints, testPoints)
    
    printResults(xPoints, yPoints, evaluationData)
    plotResults(xPoints, yPoints, evaluationData)

if __name__ == "__main__":
    main()
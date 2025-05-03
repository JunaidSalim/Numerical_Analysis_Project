import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple, List

def originalFunction(x: float) -> float:
    """Original function f(x) = 1/(1 + x²) (Runge function)"""
    return 1/(1 + x**2)

def lagrangeInterpolation(
    xPoints: np.ndarray,
    yPoints: np.ndarray,
    x: float
) -> float:
    """
    Compute Lagrange interpolation polynomial at point x.
    
    Args:
        xPoints: Array of x coordinates
        yPoints: Array of y coordinates
        x: Point at which to evaluate the polynomial
        
    Returns:
        Value of interpolation polynomial at x
    """
    n = len(xPoints)
    result = 0.0
    
    for j in range(n):
        # Compute basis polynomial L_j(x)
        basis = 1.0
        for i in range(n):
            if i != j:
                basis *= (x - xPoints[i]) / (xPoints[j] - xPoints[i])
        result += yPoints[j] * basis
        
    return result

def evaluatePoints(
    xPoints: np.ndarray,
    yPoints: np.ndarray,
    testPoints: List[float]
) -> List[dict]:
    """
    Evaluate interpolation at test points and collect results.
    
    Args:
        xPoints: Data points x coordinates
        yPoints: Data points y coordinates
        testPoints: Points at which to evaluate interpolation
        
    Returns:
        List of dictionaries containing evaluation data
    """
    evaluationData = []
    
    for x in testPoints:
        yInterp = lagrangeInterpolation(xPoints, yPoints, x)
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
    plt.figure(figsize=(12, 5))
    
    # Plot interpolation polynomial
    plt.subplot(1, 2, 1)
    xSmooth = np.linspace(min(xPoints), max(xPoints), 1000)
    ySmooth = [lagrangeInterpolation(xPoints, yPoints, x) for x in xSmooth]
    
    plt.plot(xSmooth, ySmooth, 'b-', label='Interpolation Polynomial')
    plt.plot(xPoints, yPoints, 'ro', label='Data Points')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Lagrange Interpolation')
    plt.grid(True)
    plt.legend()
    
    # Plot comparison with original function
    plt.subplot(1, 2, 2)
    ySmoothTrue = [originalFunction(x) for x in xSmooth]
    
    plt.plot(xSmooth, ySmoothTrue, 'g-', label='Original Function')
    plt.plot(xSmooth, ySmooth, 'b--', label='Interpolation Polynomial')
    plt.plot(xPoints, yPoints, 'ro', label='Data Points')
    
    # Add test points
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
    print("\nLagrange Interpolation Results:")
    print("Original Data Points:")
    print("| Index |     x     |     y     |")
    print("|-------|-----------|-----------|")
    
    for i, (x, y) in enumerate(zip(xPoints, yPoints)):
        print(f"| {i:5d} | {x:9.4f} | {y:9.4f} |")
        
    print("\nInterpolation at Test Points:")
    print("|    x     | Interpolated |  Function   |   Error    |")
    print("|----------|--------------|-------------|------------|")
    
    for data in evaluationData:
        print(f"| {data['x']:8.4f} | {data['yInterp']:10.4f} | {data['yTrue']:9.4f} | {data['error']:8.2e} |")

def main():
    try:
        # Generate more data points (15 points) over [-5, 5]
        xPoints = np.linspace(-5, 5, 15)
        yPoints = np.array([originalFunction(x) for x in xPoints])
        
        # Define test points (avoiding data points)
        testPoints = np.array([-4.2, -2.7, -1.3, 0.8, 2.4, 3.9])
        
        # Evaluate interpolation at test points
        evaluationData = evaluatePoints(xPoints, yPoints, testPoints)
        
        # Print and plot results
        printResults(xPoints, yPoints, evaluationData)
        plotResults(xPoints, yPoints, evaluationData)
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
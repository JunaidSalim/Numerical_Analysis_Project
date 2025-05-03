import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple, List

def originalFunction(x: float) -> float:
    """Original function f(x) = 1/(1 + x²) (Runge function)"""
    return 1/(1 + x**2)

def originalDerivative(x: float) -> float:
    """Derivative of the original function f'(x) = -2x/(1 + x²)²"""
    return -2*x/(1 + x**2)**2

def hermiteInterpolation(
    xPoints: np.ndarray,
    yPoints: np.ndarray,
    dydxPoints: np.ndarray,
    x: float
) -> float:
    """
    Compute Hermite interpolation polynomial at point x.
    
    Args:
        xPoints: Array of x coordinates
        yPoints: Array of y coordinates
        dydxPoints: Array of derivatives at x points
        x: Point at which to evaluate the polynomial
        
    Returns:
        Value of interpolation polynomial at x
    """
    n = len(xPoints)
    result = 0.0
    
    for i in range(n):
        # Compute L_i(x)
        Li = 1.0
        dLi = 0.0
        
        for j in range(n):
            if i != j:
                Li *= (x - xPoints[j]) / (xPoints[i] - xPoints[j])
                
                # Compute derivative of L_i(x)
                temp = 1.0 / (xPoints[i] - xPoints[j])
                for k in range(n):
                    if k != i and k != j:
                        temp *= (x - xPoints[k]) / (xPoints[i] - xPoints[k])
                dLi += temp
        
        # Compute h_i(x)
        hi = (1 - 2 * (x - xPoints[i]) * dLi) * Li * Li
        
        # Compute H_i(x)
        Hi = (x - xPoints[i]) * Li * Li
        
        result += yPoints[i] * hi + dydxPoints[i] * Hi
        
    return result

def evaluatePoints(
    xPoints: np.ndarray,
    yPoints: np.ndarray,
    dydxPoints: np.ndarray,
    testPoints: List[float]
) -> List[dict]:
    """
    Evaluate interpolation at test points and collect results.
    
    Args:
        xPoints: Data points x coordinates
        yPoints: Data points y coordinates
        dydxPoints: Derivatives at data points
        testPoints: Points at which to evaluate interpolation
        
    Returns:
        List of dictionaries containing evaluation data
    """
    evaluationData = []
    
    for x in testPoints:
        yInterp = hermiteInterpolation(xPoints, yPoints, dydxPoints, x)
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
    dydxPoints: np.ndarray,
    evaluationData: List[dict]
) -> None:
    """
    Plot the interpolation results
    """
    plt.figure(figsize=(12, 5))
    
    # Plot interpolation polynomial
    plt.subplot(1, 2, 1)
    xSmooth = np.linspace(min(xPoints), max(xPoints), 1000)
    ySmooth = [hermiteInterpolation(xPoints, yPoints, dydxPoints, x) for x in xSmooth]
    
    plt.plot(xSmooth, ySmooth, 'b-', label='Hermite Polynomial')
    plt.plot(xPoints, yPoints, 'ro', label='Data Points')
    
    # Plot derivative vectors
    for x, y, dydx in zip(xPoints, yPoints, dydxPoints):
        dx = 0.2  # Small x increment for derivative visualization
        dy = dydx * dx
        plt.arrow(x, y, dx, dy, head_width=0.05, head_length=0.1, fc='g', ec='g')
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Hermite Interpolation')
    plt.grid(True)
    plt.legend()
    
    # Plot comparison with original function
    plt.subplot(1, 2, 2)
    ySmoothTrue = [originalFunction(x) for x in xSmooth]
    
    plt.plot(xSmooth, ySmoothTrue, 'g-', label='Original Function')
    plt.plot(xSmooth, ySmooth, 'b--', label='Hermite Polynomial')
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
    dydxPoints: np.ndarray,
    evaluationData: List[dict]
) -> None:
    """
    Print interpolation results in tabular format
    """
    print("\nHermite Interpolation Results:")
    print("Original Data Points and Derivatives:")
    print("| Index |     x     |     y     |    dy/dx   |")
    print("|-------|-----------|-----------|------------|")
    
    for i, (x, y, dydx) in enumerate(zip(xPoints, yPoints, dydxPoints)):
        print(f"| {i:5d} | {x:9.4f} | {y:9.4f} | {dydx:10.4f} |")
        
    print("\nInterpolation at Test Points:")
    print("|    x     | Interpolated |  Function   |   Error    |")
    print("|----------|--------------|-------------|------------|")
    
    for data in evaluationData:
        print(f"| {data['x']:8.4f} | {data['yInterp']:10.4f} | {data['yTrue']:9.4f} | {data['error']:8.2e} |")

def main():
    try:
        # Generate data points (15 points) over [-5, 5]
        xPoints = np.linspace(-5, 5, 15)
        yPoints = np.array([originalFunction(x) for x in xPoints])
        dydxPoints = np.array([originalDerivative(x) for x in xPoints])
        
        # Define test points (avoiding data points)
        testPoints = np.array([-4.2, -3.1, -1.8, -0.7, 0.8, 2.3, 3.5, 4.4])
        
        # Evaluate interpolation at test points
        evaluationData = evaluatePoints(xPoints, yPoints, dydxPoints, testPoints)
        
        # Print and plot results
        printResults(xPoints, yPoints, dydxPoints, evaluationData)
        plotResults(xPoints, yPoints, dydxPoints, evaluationData)
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
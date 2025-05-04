import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Callable, List, Tuple

def forwardDifference(f: Callable[[float], float], x: float, h: float) -> float:
    """
    Forward difference approximation of first derivative.
    f'(x) ≈ [f(x + h) - f(x)] / h
    """
    return (f(x + h) - f(x)) / h

def backwardDifference(f: Callable[[float], float], x: float, h: float) -> float:
    """
    Backward difference approximation of first derivative.
    f'(x) ≈ [f(x) - f(x - h)] / h
    """
    return (f(x) - f(x - h)) / h

def centralDifference(f: Callable[[float], float], x: float, h: float) -> float:
    """
    Central difference approximation of first derivative.
    f'(x) ≈ [f(x + h) - f(x - h)] / (2h)
    """
    return (f(x + h) - f(x - h)) / (2 * h)

def threePointEndpoint(f: Callable[[float], float], x: float, h: float) -> float:
    """
    Three-point endpoint formula for first derivative.
    f'(x) ≈ [-3f(x) + 4f(x + h) - f(x + 2h)] / (2h)
    """
    return (-3*f(x) + 4*f(x + h) - f(x + 2*h)) / (2 * h)

def fivePointMidpoint(f: Callable[[float], float], x: float, h: float) -> float:
    """
    Five-point midpoint formula for first derivative.
    f'(x) ≈ [f(x - 2h) - 8f(x - h) + 8f(x + h) - f(x + 2h)] / (12h)
    """
    return (f(x - 2*h) - 8*f(x - h) + 8*f(x + h) - f(x + 2*h)) / (12 * h)

def compareMethods(f: Callable[[float], float], 
                  df: Callable[[float], float], 
                  x: float,
                  h: float) -> pd.DataFrame:
    """
    Compare different numerical differentiation methods with analytical solution.
    """
    trueDerivative = df(x)
    methods = {
        'Forward Difference': forwardDifference(f, x, h),
        'Backward Difference': backwardDifference(f, x, h),
        'Central Difference': centralDifference(f, x, h),
        'Three-Point Endpoint': threePointEndpoint(f, x, h),
        'Five-Point Midpoint': fivePointMidpoint(f, x, h)
    }
    
    results = []
    for methodName, approximation in methods.items():
        absoluteError = abs(approximation - trueDerivative)
        relativeError = absoluteError / abs(trueDerivative) if trueDerivative != 0 else float('inf')
        results.append({
            'Method': methodName,
            'Approximation': approximation,
            'Absolute Error': absoluteError,
            'Relative Error': relativeError
        })
    
    return pd.DataFrame(results)

def plotApproximations(f: Callable[[float], float], 
                      df: Callable[[float], float], 
                      xRange: Tuple[float, float],
                      h: float,
                      numPoints: int = 100) -> None:
    """
    Plot different numerical approximations against the analytical derivative.
    """
    x = np.linspace(xRange[0], xRange[1], numPoints)
    trueDeriv = [df(xi) for xi in x]
    
    # Calculate approximations for all methods
    forward = [forwardDifference(f, xi, h) for xi in x]
    backward = [backwardDifference(f, xi, h) for xi in x]
    central = [centralDifference(f, xi, h) for xi in x]
    threePoint = [threePointEndpoint(f, xi, h) for xi in x]
    fivePoint = [fivePointMidpoint(f, xi, h) for xi in x]
    
    plt.figure(figsize=(12, 8))
    plt.plot(x, trueDeriv, 'k-', label='Analytical', linewidth=2)
    plt.plot(x, forward, 'r--', label='Forward Difference')
    plt.plot(x, backward, 'm--', label='Backward Difference')
    plt.plot(x, central, 'b--', label='Central Difference')
    plt.plot(x, threePoint, 'g--', label='Three-Point Endpoint')
    plt.plot(x, fivePoint, 'y--', label='Five-Point Midpoint')
    
    plt.grid(True)
    plt.xlabel('x')
    plt.ylabel("f'(x)")
    plt.title('Comparison of Numerical Differentiation Methods')
    plt.legend()
    plt.show()

def main():
    print("Numerical Differentiation Methods Demonstration")
    
    # Example function: f(x) = ln(x^2 + 1) + 2*ln(x + 2)
    # Its derivative: f'(x) = (2x)/(x^2 + 1) + 2/(x + 2)
    f = lambda x: np.log(x**2 + 1) + 2*np.log(x + 2)
    df = lambda x: (2*x)/(x**2 + 1) + 2/(x + 2)
    
    x0 = 1.5  # Point of evaluation - changed to be more suitable for log function
    h = 0.01  # Step size
    
    # Compare methods
    print(f"\nComparing methods at x = {x0:.4f} with h = {h}")
    results = compareMethods(f, df, x0, h)
    print("\nResults:")
    print(results.to_string(index=False, float_format=lambda x: '{:.10f}'.format(x)))
    
    # Plot comparisons
    plotApproximations(f, df, (0.1, 4), h)  # Adjusted range to avoid x ≤ 0 for log function
    
    # Study effect of step size
    hValues = [0.1, 0.01, 0.001, 0.0001, 0.00001]
    errorData = []
    
    for h in hValues:
        centralApprox = centralDifference(f, x0, h)
        trueValue = df(x0)
        error = abs(centralApprox - trueValue)
        errorData.append({
            'Step Size (h)': h,
            'Approximation': centralApprox,
            'True Value': trueValue,
            'Absolute Error': error
        })
    
    print("\nEffect of Step Size on Central Difference Method:")
    errorDf = pd.DataFrame(errorData)
    print(errorDf.to_string(index=False, float_format=lambda x: '{:.10f}'.format(x)))
    
    # Plot error vs step size
    plt.figure(figsize=(10, 6))
    plt.loglog(hValues, [d['Absolute Error'] for d in errorData], 'bo-')
    plt.grid(True)
    plt.xlabel('Step Size (h)')
    plt.ylabel('Absolute Error')
    plt.title('Error Analysis: Step Size vs Absolute Error')
    plt.show()
    
    print("\nProgram completed successfully")

if __name__ == "__main__":
    main()
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Callable, List, Tuple

def trapezoidalRule(f: Callable[[float], float], a: float, b: float, n: int) -> float:
    """
    Implement the Trapezoidal Rule for numerical integration.
    
    Parameters:
    f (callable): Function to integrate
    a (float): Lower bound
    b (float): Upper bound
    n (int): Number of subintervals
    
    Returns:
    float: Approximate integral value
    """
    h = (b - a) / n
    x = np.linspace(a, b, n+1)
    y = f(x)
    
    return h * (0.5 * y[0] + np.sum(y[1:-1]) + 0.5 * y[-1])

def simpsonsRule(f: Callable[[float], float], a: float, b: float, n: int) -> float:
    """
    Implement Simpson's 1/3 Rule for numerical integration.
    
    Parameters:
    f (callable): Function to integrate
    a (float): Lower bound
    b (float): Upper bound
    n (int): Number of subintervals (must be even)
    
    Returns:
    float: Approximate integral value
    """
    if n % 2 != 0:
        raise ValueError("n must be even for Simpson's rule")
    
    h = (b - a) / n
    x = np.linspace(a, b, n+1)
    y = f(x)
    
    return h/3 * (y[0] + 4*np.sum(y[1:-1:2]) + 2*np.sum(y[2:-1:2]) + y[-1])

def analyzeConvergence(f: Callable[[float], float], 
                      exactIntegral: float,
                      a: float, 
                      b: float, 
                      maxN: int = 100) -> pd.DataFrame:
    """
    Analyze convergence of both integration methods.
    """
    nValues = [2**i for i in range(2, int(np.log2(maxN))+1)]
    results = []
    
    for n in nValues:
        trapError = abs(trapezoidalRule(f, a, b, n) - exactIntegral)
        simpError = abs(simpsonsRule(f, a, b, n) - exactIntegral)
        
        results.append({
            'Subintervals': n,
            'Trapezoidal Error': trapError,
            'Simpson Error': simpError
        })
    
    return pd.DataFrame(results)

def plotErrorConvergence(f: Callable[[float], float], 
                        exactIntegral: float,
                        a: float, 
                        b: float, 
                        maxN: int = 100) -> None:
    """
    Plot error convergence for both methods.
    """
    nValues = [2**i for i in range(2, int(np.log2(maxN))+1)]
    trapErrors = []
    simpErrors = []
    
    for n in nValues:
        trapErrors.append(abs(trapezoidalRule(f, a, b, n) - exactIntegral))
        simpErrors.append(abs(simpsonsRule(f, a, b, n) - exactIntegral))
    
    plt.figure(figsize=(10, 6))
    plt.loglog(nValues, trapErrors, 'bo-', label='Trapezoidal Rule')
    plt.loglog(nValues, simpErrors, 'ro-', label="Simpson's Rule")
    plt.grid(True)
    
    plt.xticks(nValues, [str(n) for n in nValues])
    
    plt.xlabel('Number of Subintervals (n)')
    plt.ylabel('Absolute Error (log scale)')
    plt.title('Error Convergence of Numerical Integration Methods')
    plt.legend()
    plt.show()

def visualizeIntegration(f: Callable[[float], float], 
                        a: float, 
                        b: float, 
                        n: int) -> None:
    """
    Visualize the function and the approximation using both methods.
    """
    x = np.linspace(a, b, 200)
    y = f(x)
    
    xn = np.linspace(a, b, n+1)
    yn = f(xn)
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(121)
    plt.plot(x, y, 'b-', label='Function')
    plt.fill_between(xn, 0, yn, alpha=0.3, color='lightblue', label='Area')
    plt.plot(xn, yn, 'ro', label='Sampling Points')
    for i in range(n):
        plt.plot([xn[i], xn[i+1]], [yn[i], yn[i+1]], 'r-', linewidth=1, label='Trapezoids' if i == 0 else "")
    plt.grid(True)
    plt.title('Trapezoidal Rule')
    plt.legend()
    
    plt.subplot(122)
    plt.plot(x, y, 'b-', label='Function')
    plt.fill_between(x, 0, y, alpha=0.3, color='lightblue', label='Area')
    plt.plot(xn, yn, 'ro', label='Sampling Points')
    
    for i in range(0, n-1, 2):
        x_local = np.linspace(xn[i], xn[i+2], 100)
        
        x0, x1, x2 = xn[i], xn[i+1], xn[i+2]
        y0, y1, y2 = yn[i], yn[i+1], yn[i+2]
        
        def L0(x): return ((x-x1)*(x-x2))/((x0-x1)*(x0-x2))
        def L1(x): return ((x-x0)*(x-x2))/((x1-x0)*(x1-x2))
        def L2(x): return ((x-x0)*(x-x1))/((x2-x0)*(x2-x1))
        
        y_local = y0*L0(x_local) + y1*L1(x_local) + y2*L2(x_local)
        
        plt.plot(x_local, y_local, 'r-', linewidth=1, label='Simpson Curves' if i == 0 else "")
    
    plt.grid(True)
    plt.title("Simpson's Rule")
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def main():
    print("Numerical Integration Methods Demonstration")
    
    f = lambda x: np.exp(x) - 5*x
    
    a, b = 0, 1
    n = 10
    
    exactIntegral = (np.exp(b) - 5*b**2/2) - (np.exp(a) - 5*a**2/2)
    
    trapResult = trapezoidalRule(f, a, b, n)
    simpResult = simpsonsRule(f, a, b, n)
    
    print(f"\nResults with {n} subintervals:")
    print(f"Trapezoidal Rule: {trapResult:.10f}")
    print(f"Simpson's Rule:   {simpResult:.10f}")
    print(f"Exact Value:      {exactIntegral:.10f}")
    
    print("\nConvergence Analysis:")
    results = analyzeConvergence(f, exactIntegral, a, b)
    print(results.to_string(index=False, float_format=lambda x: '{:.10e}'.format(x)))
    
    visualizeIntegration(f, a, b, n)
    
    plotErrorConvergence(f, exactIntegral, a, b)
    
    print("\nProgram completed successfully")

if __name__ == "__main__":
    main()
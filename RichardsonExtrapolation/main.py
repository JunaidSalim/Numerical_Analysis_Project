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

def richardsonExtrapolation(f: Callable[[float], float], x: float, h: float, n: int) -> float:
    """
    Richardson Extrapolation to improve accuracy of numerical differentiation.
    """
    D = np.zeros((n, n))
    
    for i in range(n):
        hi = h / (2**i)
        D[i, 0] = forwardDifference(f, x, hi)
    
    for j in range(1, n):
        for i in range(n-j):
            power = 2**(j)
            D[i, j] = (power * D[i+1, j-1] - D[i, j-1]) / (power - 1)
    
    return D[0, n-1]

def computeRichardsonTable(f: Callable[[float], float], x: float, h: float, n: int) -> pd.DataFrame:
    """
    Compute and display the full Richardson extrapolation table.
    """
    D = np.zeros((n, n))
    
    for i in range(n):
        hi = h / (2**i)
        D[i, 0] = forwardDifference(f, x, hi)
    
    for j in range(1, n):
        for i in range(n-j):
            power = 2**(j)
            D[i, j] = (power * D[i+1, j-1] - D[i, j-1]) / (power - 1)
    
    df = pd.DataFrame(D)
    df.columns = [f'O(h^{i+1})' for i in range(n)]
    df.index = [f'h/{2**i}' for i in range(n)]
    return df

def plotLevelVsError(f: Callable[[float], float], df: Callable[[float], float],
                    x: float, h: float, nLevels: int) -> None:
    """
    Plot relationship between Richardson levels and error.
    """
    trueValue = df(x)
    levels = list(range(1, nLevels + 1))
    errors = []
    
    for level in levels:
        approx = richardsonExtrapolation(f, x, h, level)
        error = abs(approx - trueValue)
        errors.append(error)
    
    plt.figure(figsize=(10, 6))
    plt.semilogy(levels, errors, 'go-', linewidth=2, markersize=8)
    plt.grid(True)
    plt.xlabel('Richardson Level')
    plt.ylabel('Absolute Error (log scale)')
    plt.title('Error vs Richardson Extrapolation Level')
    
    for i, error in enumerate(errors):
        plt.annotate(f'{error:.2e}', 
                    (levels[i], error),
                    textcoords="offset points",
                    xytext=(0,10),
                    ha='center')
    
    plt.show()

def main():
    print("Richardson Extrapolation Demonstration")
    
    f = lambda x: np.log(x**2 + 1) + 2*np.log(x + 2)
    df = lambda x: (2*x)/(x**2 + 1) + 2/(x + 2)
    
    x0 = 1.5
    h = 0.1
    nLevels = 5
    
    print("\nRichardson Extrapolation Table:")
    richTable = computeRichardsonTable(f, x0, h, nLevels)
    print(richTable.to_string(float_format=lambda x: '{:.10f}'.format(x)))
    
    trueValue = df(x0)
    print(f"\nTrue derivative at x = {x0:.4f}: {trueValue:.10f}")
    
    plotLevelVsError(f, df, x0, h, nLevels)
    
    print("\nImprovement with Increasing Richardson Levels:")
    for i in range(1, nLevels + 1):
        approx = richardsonExtrapolation(f, x0, h, i)
        error = abs(approx - trueValue)
        print(f"Level {i}: Approximation = {approx:.10f}, Error = {error:.10e}")
    
    print("\nProgram completed successfully")

if __name__ == "__main__":
    main()
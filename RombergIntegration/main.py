import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Callable, List, Tuple

def trapezoidalRule(f: Callable[[float], float], a: float, b: float, n: int) -> float:
    """
    Compute trapezoidal rule approximation with n subintervals.
    """
    h = (b - a) / n
    x = np.linspace(a, b, n+1)
    y = f(x)
    return h * (0.5 * y[0] + np.sum(y[1:-1]) + 0.5 * y[-1])

def rombergIntegration(f: Callable[[float], float], a: float, b: float, maxLevel: int) -> np.ndarray:
    """
    Implement Romberg Integration method.
    
    Parameters:
    f (callable): Function to integrate
    a (float): Lower bound
    b (float): Upper bound
    maxLevel (int): Maximum level of refinement
    
    Returns:
    np.ndarray: Romberg table
    """
    R = np.zeros((maxLevel, maxLevel))
    
    # Fill first column with trapezoidal rule values
    for i in range(maxLevel):
        n = 2**i
        R[i, 0] = trapezoidalRule(f, a, b, n)
    
    # Fill rest of the table using Richardson extrapolation
    for j in range(1, maxLevel):
        for i in range(maxLevel - j):
            power = 4**j
            R[i, j] = (power * R[i+1, j-1] - R[i, j-1]) / (power - 1)
    
    return R

def displayRombergTable(R: np.ndarray) -> None:
    """
    Display Romberg Integration table in a formatted way.
    """
    n = len(R)
    df = pd.DataFrame(R)
    df.columns = [f'R(i,{j})' for j in range(n)]
    df.index = [f'n={2**i}' for i in range(n)]
    return df

def plotConvergence(R: np.ndarray, exactValue: float) -> None:
    """
    Plot convergence of Romberg Integration method and show minimum error at each level.
    """
    levels = range(len(R))
    errors = [abs(R[0, j] - exactValue) for j in levels]
    
    # Calculate minimum error at each level
    min_errors = []
    for j in levels:
        level_errors = [abs(R[i, j] - exactValue) for i in range(len(R)-j)]
        min_error = min(level_errors)
        min_errors.append(min_error)
    
    plt.figure(figsize=(10, 6))
    # Plot both standard errors and minimum errors
    plt.semilogy(levels, errors, 'go-', linewidth=2, markersize=8, label='Standard Error')
    plt.semilogy(levels, min_errors, 'bo-', linewidth=2, markersize=8, label='Minimum Error')
    plt.grid(True)
    plt.xlabel('Romberg Level')
    plt.ylabel('Absolute Error (log scale)')
    plt.title('Romberg Integration Convergence')
    plt.legend()
    
    # Add error values as annotations
    for i, (error, min_error) in enumerate(zip(errors, min_errors)):
        plt.annotate(f'Std: {error:.2e}\nMin: {min_error:.2e}', 
                    (i, error),
                    textcoords="offset points",
                    xytext=(0,10),
                    ha='center',
                    va='bottom')
    
    plt.show()

def visualizeIntegration(f: Callable[[float], float], a: float, b: float, n: int) -> None:
    """
    Visualize the function and integration approximation.
    """
    x = np.linspace(a, b, 200)
    y = f(x)
    
    # Points for visualization
    xn = np.linspace(a, b, n+1)
    yn = f(xn)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'b-', label='Function')
    plt.fill_between(x, 0, y, alpha=0.3, color='lightblue', label='Area')
    plt.plot(xn, yn, 'ro', label='Sampling Points')
    
    # Connect points with lines to show trapezoids
    for i in range(n):
        plt.plot([xn[i], xn[i+1]], [yn[i], yn[i+1]], 'r-', linewidth=1, 
                label='Trapezoids' if i == 0 else "")
    
    plt.grid(True)
    plt.title('Romberg Integration Base (Trapezoidal Rule)')
    plt.legend()
    plt.show()

def main():
    print("Romberg Integration Method Demonstration")
    
    # Test function: f(x) = e^x - 5x
    # Analytical integral: e^x - 5x²/2
    f = lambda x: np.exp(x) - 5*x
    
    # Integration limits [0, 1]
    a, b = 0, 1
    maxLevel = 5
    
    # Compute exact integral: [e^b - 5b²/2] - [e^a - 5a²/2]
    exactIntegral = (np.exp(b) - 5*b**2/2) - (np.exp(a) - 5*a**2/2)
    
    # Compute Romberg table
    R = rombergIntegration(f, a, b, maxLevel)
    
    # Display results
    print("\nRomberg Integration Table:")
    rombergTable = displayRombergTable(R)
    print(rombergTable.to_string(float_format=lambda x: '{:.10f}'.format(x)))
    
    print(f"\nExact Value: {exactIntegral:.10f}")
    print(f"Best Approximation (R(0,{maxLevel-1})): {R[0, maxLevel-1]:.10f}")
    print(f"Absolute Error: {abs(R[0, maxLevel-1] - exactIntegral):.2e}")
    
    # Display minimum errors at each level
    print("\nError Analysis by Level:")
    for j in range(maxLevel):
        level_errors = [abs(R[i, j] - exactIntegral) for i in range(maxLevel-j)]
        min_error = min(level_errors)
        min_error_index = level_errors.index(min_error)
        print(f"Level {j}:")
        print(f"  Standard Error (R(0,{j})): {abs(R[0, j] - exactIntegral):.2e}")
        print(f"  Minimum Error (R({min_error_index},{j})): {min_error:.2e}")
    
    # Visualize the base approximation
    visualizeIntegration(f, a, b, 8)  # Using 8 subintervals for clear visualization
    
    # Plot convergence with minimum errors
    plotConvergence(R, exactIntegral)
    
    print("\nProgram completed successfully")

if __name__ == "__main__":
    main()
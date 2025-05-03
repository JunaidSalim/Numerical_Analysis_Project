import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple

def newtonRaphson(
    f: Callable[[float], float],
    df: Callable[[float], float],
    x0: float,
    tolerance: float = 1e-6,
    maxIterations: int = 100
) -> Tuple[float, int, list]:
    """
    Implementation of the Newton-Raphson method for finding roots of a function.
    
    Args:
        f: The function whose root we want to find
        df: The derivative of the function f
        x0: Initial guess
        tolerance: Stopping tolerance
        maxIterations: Maximum number of iterations allowed
        
    Returns:
        Tuple containing:
        - The approximate root
        - Number of iterations taken
        - List of iteration data
    """
    iterationData = []
    iteration = 0
    x_current = x0
    error = float('inf')
    
    while iteration < maxIterations and error > tolerance:
        f_value = f(x_current)
        df_value = df(x_current)
        
        if abs(df_value) < 1e-10:  # Avoid division by near-zero
            raise ValueError("Derivative too close to zero")
            
        x_next = x_current - f_value/df_value
        error = abs(x_next - x_current)
        
        iterationData.append({
            'iteration': iteration + 1,
            'x': x_next,
            'f(x)': f(x_next),
            'error': error
        })
        
        x_current = x_next
        iteration += 1
    
    return x_current, iteration, iterationData

def plotResults(f: Callable[[float], float], df: Callable[[float], float], 
                iterationData: list, x_range: tuple = (-5, 5)) -> None:
    """
    Plot the function, its derivative, and convergence behavior.
    """
    plt.figure(figsize=(15, 5))
    
    # Plot error vs iterations
    plt.subplot(1, 3, 1)
    iterations = [data['iteration'] for data in iterationData]
    errors = [data['error'] for data in iterationData]
    plt.plot(iterations, errors, 'b-o')
    plt.xlabel('Iterations')
    plt.ylabel('Error')
    plt.title('Error vs Iterations')
    plt.grid(True)
    
    # Plot function
    plt.subplot(1, 3, 2)
    x = np.linspace(x_range[0], x_range[1], 1000)
    y = [f(val) for val in x]
    plt.plot(x, y, 'r-', label='f(x)')
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Function')
    plt.legend()
    plt.grid(True)
    
    # Plot derivative
    plt.subplot(1, 3, 3)
    y_derivative = [df(val) for val in x]
    plt.plot(x, y_derivative, 'g-', label="f'(x)")
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    plt.xlabel('x')
    plt.ylabel("f'(x)")
    plt.title('Derivative')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def printResults(iterationData: list, totalIterations: int) -> None:
    """
    Print the results in tabular format with detailed information about each iteration.
    """
    print("\nNewton-Raphson Method Results:")
    print("| Iteration |    x_n    |   f(x_n)  |   Error   |")
    print("|-----------|-----------|-----------|-----------|")
    
    for data in iterationData:
        print(f"| {data['iteration']:9d} | {data['x']:9.6f} | {data['f(x)']:9.6f} | {data['error']:9.6f} |")
    
    print(f"\nTotal Iterations: {totalIterations}")

def main():
    # Example: Find the root of f(x) = e^x - 5x
    def f(x: float) -> float:
        return np.exp(x) - 5*x
    
    def df(x: float) -> float:
        return np.exp(x) - 5
    
    try:
        x0 = 2.0  # Initial guess
        root, iterations, data = newtonRaphson(f, df, x0)
        print(f"\nTest Case: Finding root of f(x) = e^x - 5x")
        print(f"Initial guess x0 = {x0}")
        printResults(data, iterations)
        plotResults(f, df, data, x_range=(-1, 3))  # Adjusted range for better visualization
        print(f"\nRoot found: {root:.8f}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple

def secant(
    f: Callable[[float], float],
    x0: float,
    x1: float,
    tolerance: float = 1e-6,
    maxIterations: int = 100
) -> Tuple[float, int, list]:
    """
    Implementation of the Secant method for finding roots of a function.
    
    Args:
        f: The function whose root we want to find
        x0: First initial guess
        x1: Second initial guess
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
    x_prev = x0
    x_curr = x1
    error = float('inf')
    
    while iteration < maxIterations and error > tolerance:
        f_prev = f(x_prev)
        f_curr = f(x_curr)
        
        if abs(f_curr - f_prev) < 1e-10:  # Avoid division by near-zero
            raise ValueError("Secant slope too close to zero")
            
        # Calculate next approximation using secant formula
        x_next = x_curr - f_curr * (x_curr - x_prev) / (f_curr - f_prev)
        error = abs(x_next - x_curr)
        
        iterationData.append({
            'iteration': iteration + 1,
            'x': x_next,
            'f(x)': f(x_next),
            'error': error
        })
        
        x_prev = x_curr
        x_curr = x_next
        iteration += 1
    
    return x_curr, iteration, iterationData

def plotResults(f: Callable[[float], float], iterationData: list, 
                x_range: tuple = (-5, 5)) -> None:
    """
    Plot the function and convergence behavior.
    """
    plt.figure(figsize=(12, 5))
    
    # Plot error vs iterations
    plt.subplot(1, 2, 1)
    iterations = [data['iteration'] for data in iterationData]
    errors = [data['error'] for data in iterationData]
    plt.plot(iterations, errors, 'b-o')
    plt.xlabel('Iterations')
    plt.ylabel('Error')
    plt.title('Error vs Iterations')
    plt.grid(True)
    
    # Plot function
    plt.subplot(1, 2, 2)
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
    
    plt.tight_layout()
    plt.show()

def printResults(iterationData: list, totalIterations: int) -> None:
    """
    Print the results in tabular format with detailed information about each iteration.
    """
    print("\nSecant Method Results:")
    print("| Iteration |    x_n    |   f(x_n)  |   Error   |")
    print("|-----------|-----------|-----------|-----------|")
    
    for data in iterationData:
        print(f"| {data['iteration']:9d} | {data['x']:9.6f} | {data['f(x)']:9.6f} | {data['error']:9.6f} |")
    
    print(f"\nTotal Iterations: {totalIterations}")

def main():
    # Example: Find the root of f(x) = e^x - 5x
    def f(x: float) -> float:
        return np.exp(x) - 5*x
    
    try:
        x0, x1 = 1.0, 2.0  # Initial guesses bracketing the root near 1.6
        root, iterations, data = secant(f, x0, x1)
        print(f"\nTest Case: Finding root of f(x) = e^x - 5x")
        print(f"Initial guesses x0 = {x0}, x1 = {x1}")
        printResults(data, iterations)
        plotResults(f, data, x_range=(-1, 3))  # Same range as used in Newton-Raphson
        print(f"\nRoot found: {root:.8f}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
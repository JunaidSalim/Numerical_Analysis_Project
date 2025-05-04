import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple

def fixedPointIteration(
    g: Callable[[float], float],
    x0: float,
    tolerance: float = 1e-6,
    maxIterations: int = 100
) -> Tuple[float, int, list]:
    """
    Implementation of the Fixed Point Iteration method for finding fixed points of a function.
    
    Args:
        g: The iteration function g(x) where x = g(x) at the fixed point
        x0: Initial guess
        tolerance: Stopping tolerance
        maxIterations: Maximum number of iterations allowed
        
    Returns:
        Tuple containing:
        - The approximate fixed point
        - Number of iterations taken
        - List of iteration data
    """
    iterationData = []
    iteration = 0
    x_current = x0
    error = float('inf')
    
    while iteration < maxIterations and error > tolerance:
        x_next = g(x_current)
        error = abs(x_next - x_current)
        
        iterationData.append({
            'iteration': iteration + 1,
            'x': x_next,
            'g(x)': g(x_next),
            'error': error
        })
        
        x_current = x_next
        iteration += 1
    
    return x_current, iteration, iterationData

def plotResults(iterationData: list, g: Callable[[float], float]) -> None:
    """
    Plot the error vs iterations and the iteration function.
    """
    iterations = [data['iteration'] for data in iterationData]
    errors = [data['error'] for data in iterationData]
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(iterations, errors, 'b-o')
    plt.xlabel('Iterations')
    plt.ylabel('Error')
    plt.title('Error vs Iterations')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    x = np.linspace(-2, 2, 1000)
    y = [g(val) for val in x]
    plt.plot(x, y, 'r-', label='g(x)')
    plt.plot(x, x, 'k--', label='y=x')
    plt.xlabel('x')
    plt.ylabel('g(x)')
    plt.title('Iteration Function')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def printResults(iterationData: list, totalIterations: int) -> None:
    """
    Print the results in tabular format with detailed information about each iteration.
    """
    print("\nFixed Point Iteration Results:")
    print("| Iteration |    x_n    |   g(x_n)  |   Error   |")
    print("|-----------|-----------|-----------|-----------|")
    
    for data in iterationData:
        print(f"| {data['iteration']:9d} | {data['x']:9.6f} | {data['g(x)']:9.6f} | {data['error']:9.6f} |")
    
    print(f"\nTotal Iterations: {totalIterations}")

def main():
    def g(x: float) -> float:
        return (x + 1/x)/2
    
    x0 = 2
    root, iterations, data = fixedPointIteration(g, x0)
    print(f"\nTest Case: Finding fixed point of g(x) = (x + 1/x)/2")
    print(f"Initial guess x0 = {x0}")
    printResults(data, iterations)
    plotResults(data, g)
    print(f"\nFixed point found: {root:.8f}")

if __name__ == "__main__":
    main()
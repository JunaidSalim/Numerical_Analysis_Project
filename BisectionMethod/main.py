import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple

def bisectionMethod(
    function: Callable[[float], float],
    lowerBound: float,
    upperBound: float,
    tolerance: float = 1e-6,
    maxIterations: int = 100
) -> Tuple[float, int, list]:
    """
    Implementation of the bisection method for finding roots of a function.
    Stops when either the error is less than tolerance or max iterations is reached.
    
    Args:
        function: The function to find the root of
        lowerBound: Lower bound of the interval
        upperBound: Upper bound of the interval
        tolerance: Stopping tolerance
        maxIterations: Maximum number of iterations allowed
        
    Returns:
        Tuple containing:
        - The approximate root
        - Number of iterations taken
        - List of iteration data
    """
    if function(lowerBound) * function(upperBound) >= 0:
        raise ValueError("Function must have opposite signs at the interval endpoints")
    
    iterationData = []
    iteration = 0
    error = float('inf')
    previousMidpoint = None
    
    while iteration < maxIterations and error > tolerance:
        midpoint = (lowerBound + upperBound) / 2
        fMidpoint = function(midpoint)
        
        if previousMidpoint is not None:
            error = abs(midpoint - previousMidpoint)
        else:
            error = float('inf')
        
        iterationData.append({
            'iteration': iteration + 1,
            'x': midpoint,
            'f(x)': fMidpoint,
            'error': error
        })
            
        if function(lowerBound) * fMidpoint < 0:
            upperBound = midpoint
        else:
            lowerBound = midpoint
            
        previousMidpoint = midpoint
        iteration += 1
    
    return midpoint, iteration, iterationData

def plotResults(iterationData: list) -> None:
    """
    Plot the error vs iterations and the function curve.
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
    x = np.linspace(0, 10, 1000)
    y = np.cos(x) - x
    plt.plot(x, y, 'r-')
    plt.axhline(y=0, color='k', linestyle='--')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Function Curve')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def printResults(iterationData: list, totalIterations: int, a: float, b: float, function: Callable[[float], float]) -> None:
    """
    Print the results in tabular format with detailed information about each iteration.
    """
    print("\nBisection Method Results:")
    print("| Iteration | a        | f(a)      | b        | f(b)      | x        | f(x)      | Error     |")
    print("|-----------|----------|-----------|----------|-----------|----------|-----------|-----------|")
    
    previous_x = None
    for data in iterationData:
        current_x = data['x']
        error = abs(current_x - previous_x) if previous_x is not None else float('inf')
        
        print(f"| {data['iteration']:9d} | "
              f"{a:.6f} | {function(a):.6f} | "
              f"{b:.6f} | {function(b):.6f} | "
              f"{current_x:.6f} | {data['f(x)']:.6f} | "
              f"{error:.6f} |")
        
        if function(a) * data['f(x)'] < 0:
            b = current_x
        else:
            a = current_x
            
        previous_x = current_x
    
    print(f"\nTotal Iterations: {totalIterations}")

def main():
    def f(x: float) -> float:
        return np.cos(x) - x
    
    testCases = [
        (0, 1),
    ]
    
    for i, (a, b) in enumerate(testCases, 1):
        print(f"\nTest Case {i}: Interval [{a}, {b}]")
        try:
            root, iterations, data = bisectionMethod(f, a, b)
            printResults(data, iterations, a, b, f)
            plotResults(data)
        except ValueError as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
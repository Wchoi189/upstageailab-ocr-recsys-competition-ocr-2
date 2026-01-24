def factorial(n):
    """
    Calculate the factorial of a non-negative integer n.
    
    Args:
        n (int): A non-negative integer
        
    Returns:
        int: The factorial of n (n!)
        
    Raises:
        ValueError: If n is negative
        TypeError: If n is not an integer
    """
    if not isinstance(n, int):
        raise TypeError("Input must be an integer")
    
    if n < 0:
        raise ValueError("Factorial is not defined for negative numbers")
    
    if n == 0 or n == 1:
        return 1
    
    result = 1
    for i in range(2, n + 1):
        result *= i
    
    return result


# Example usage
if __name__ == "__main__":
    # Test the function
    print(f"factorial(0) = {factorial(0)}")  # Expected: 1
    print(f"factorial(1) = {factorial(1)}")  # Expected: 1
    print(f"factorial(5) = {factorial(5)}")  # Expected: 120
    print(f"factorial(10) = {factorial(10)}")  # Expected: 3628800
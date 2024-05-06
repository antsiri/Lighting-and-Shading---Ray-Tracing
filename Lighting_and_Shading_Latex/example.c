#include <stdio.h>

// Function to calculate the factorial
int factorial(int n) {
    // Base case: if n is 0 or 1, return 1
    if (n == 0 || n == 1)
        return 1;
    // Recursive case: return n times the factorial of (n-1)
    else
        return n * factorial(n - 1);
}

int main() {
    int num;
    printf("Enter a number to find its factorial: ");
    scanf("%d", &num);
    // Call the factorial function and print the result
    printf("Factorial of %d = %d\n", num, factorial(num));
    return 0;
}
def factorial(n):
  if n == 0:
    return 1
  else:
    return n*factorial(n-1)
n=102
print("the factorial of %i is %i"%(n, factorial(n)))

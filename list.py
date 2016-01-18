m=5; n=4
def List(m,n):
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    alpha = list(alphabet)
    if m > 26:
        return "beyond the limit of letters. 'm' must be 26 or less"
    else:
        return [alpha[0:m] for i in range(0, n)]
print(List(m,n))

def ListsFromTuples(x):
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    alpha = list(alphabet)
    return [(alpha[x],alpha.index(alpha[x])+1) for x in range(26)]
    

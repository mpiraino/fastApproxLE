import mpmath
mpmath.mp.dps=1000

def trace_of(A):
    d=A.rows
    return mpmath.nsum(lambda l: A[l,l], [0,d-1])

def minor(n,m,A):
    d = A.rows
    
    M=mpmath.matrix(d-1)
    
    for i in range(0,d-1):
        for j in range(0,d-1):
            if i < n and j < m:
                M[i,j] = A[i,j]
            elif i >= n and j < m:
                M[i,j] = A[i+1,j]
            elif i < n and j >= m:
                M[i,j] = A[i,j+1]
            elif i >= n and j >= m:
                M[i,j] = A[i+1,j+1]
    
    return M

def adj(A):
    d=A.rows
    M=mpmath.matrix(d)
    for i in range(0,d):
        for j in range(0,d):
            M[i,j] = ((-1) ** (i+j)) * mpmath.det(minor(j,i,A))
    
    return M

def operator_trace(A, s):
    d=A.rows
    eigenvalues = sorted(mpmath.mp.eig(A, left = False, right = False), key = abs , reverse =True)
    denominator = mpmath.nprod(lambda j : (1 - (eigenvalues[int(j)-1]/eigenvalues[0])), [2,d] )
               
    return (mpmath.chop((eigenvalues[0] ** s)/ denominator), mpmath.chop((mpmath.log(eigenvalues[0]) * (eigenvalues[0] ** s))/ denominator))


def shift(per):
    shift_per = per + per
    return shift_per[1:len(per)+1]

def cocycle(word, mathcalA):
    #computes A^(n)(x) for x a periodic point
    d=mathcalA('0').rows
    A=mpmath.eye(d)
    for i in range(len(word)):
        A = A * mathcalA(word)
        word = shift(word)
    return A

def weight(word, varphi):
    #computes $e^{S_{n}\varphi(x)} for x a periodic point
    a = 1
    for i in range(len(word)):
        a = a * varphi(word)
        word = shift(word)
    return a

def lyapunov_exponent(mathcalA, varphi, periodic_points , alg = 'basic', norm = 2):
    
    k= max([len(word) for word in periodic_points])

    if alg == 'basic':
        approx_basic = []

        for n in range(1,k+1):
            integral = sum([mpmath.log(mpmath.norm(cocycle(word, mathcalA), p = norm)) * weight(word, varphi) for word in periodic_points if len(word) == n])
            normalization = sum([weight(word, varphi) for word in periodic_points if len(word) == n])
            approx_basic.append(integral/(n * normalization))

        return approx_basic
    
    elif alg == 'pollicott':
        #Compute the operator trace for each periodic point
        op_trace = { word : operator_trace(cocycle(word, mathcalA),0)[0] for word in periodic_points}
        op_trace_der = {word : operator_trace(cocycle(word, mathcalA), 0)[1] for word in periodic_points}

        #Compute traces for products of transfer operator put in dictionary indexed by power
        trace = [sum([(op_trace[word] * weight(word, varphi)) for word in periodic_points if len(word) == n]) for n in range(1,k+1)]           
        trace_der = [sum([(op_trace_der[word] * weight(word, varphi)) for word in periodic_points if len(word) == n]) for n in range(1,k+1)]
        
        coefficients = [mpmath.mpf(1)]

        coefficients_der = [mpmath.mpf(0)]

        for n in range(1,k+1):
            M = mpmath.matrix(n)
            Der_M = mpmath.matrix(n)
            for i in range(0,n):
                for j in range(0,n):
                    if j > i+1:
                        M[i,j] = 0
                        Der_M[i,j] = 0
                    elif j == i + 1:
                        M[i,j] = n-j
                        Der_M[i,j] = 0
                    else:
                        M[i,j] = trace[i-j]
                        Der_M[i,j] = trace_der[i-j]

            coefficients.append((((-1) ** n)/mpmath.fac(n)) * mpmath.det(M))

            if n == 1:
                coefficients_der.append((((-1) ** n)/mpmath.fac(n)) * trace_der[0])
            else:
                #Use Jacobi's formula to compute derivative of coefficients
                coefficients_der.append((((-1) ** n)/mpmath.fac(n)) * trace_of(adj(M)*Der_M))
                
                
        approximation=[]

        for n in range(1,k+1):    
            approximation.append(sum([coefficients_der[m] for m in range(1,n+1)]) / sum([m * coefficients[m] for m in range(1,n+1)]))
            
        return approximation

    else:
        return "Choices of algorithm are 'basic' and 'pollicott'"



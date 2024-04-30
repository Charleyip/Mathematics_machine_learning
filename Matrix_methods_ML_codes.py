# Degree of the Vandermonde matrix to be determined by user
def vander_degree(matrix, degree):
    '''
    Generating the Vandermonde matrix for the given scalar 'matrix'
    
    '''
    vander_matrix = np.vander(matrix.flatten(), degree + 1, increasing=True)

    return vander_matrix

#basic least_squares problem     
def least_squares(A, d):
    """
    Solve the least squares problem for Ax = d.
    """
    # Compute A^T * A
    AtA = A.T @ A
    
    # Compute the inverse of A^T * A
    AtA_inv = np.linalg.inv(AtA)
    
    # Compute A^T * d
    Atd = A.T @ d
    
    # Compute the solution vector x
    x = AtA_inv @ Atd
    
    return x



#
def ItSingValThresh(Y, r):
    """
    Iterative Singular Value Thresholding function for Matrix Completion
    """
    tol = 10**(-3)  # difference between iterates at termination
    max_its = 100;
    n,p = Y.shape 
    X = np.array(Y) #make a copy so operations do not mutate the original
    X[np.isnan(X)] = 0 # Fill in missing entries with zeros

    err = 10**6 
    itt = 0
    
    while err > tol and itt < max_its:
        U,s,VT = np.linalg.svd(X, full_matrices=False)
        V, S = VT.T, np.diag(s[:r])
        Xnew = U[:, :r] @ S @ V[:r, :]
        for i in range(n):
            for j in range(p):
                if ~np.isnan(Y[i,j]):  #replace Xnew with known entries
                    Xnew[i,j] = Y[i,j]
        err = np.linalg.norm(X-Xnew,'fro') 
        X = Xnew
        itt += 1
    return X,err

def gram_schmidt(B):
    """Orthogonalize a set of vectors stored as the columns of matrix B."""
    # Get the number of vectors.
    m, n = B.shape
    # Create new matrix to hold the orthonormal basis
    U = np.zeros([m,n]) 
    for j in range(n):
        # To orthogonalize the vector in column j with respect to the
        # previous vectors, subtract from it its projection onto
        # each of the previous vectors.
        v = B[:,j].copy()
        for k in range(j):
            v -= np.dot(U[:, k], B[:, j]) * U[:, k]
        if np.linalg.norm(v)>1e-10:
            U[:, j] = v / np.linalg.norm(v)
    return U

# if __name__ == '__main__':
#     B1 = np.array([[1.0, 1.0, 0.0], [2.0, 2.0, 0.0], [2.0, 2.0, 1.0]])
#     A1 = gram_schmidt(B1)
#     print(A1)
#     A2 = gram_schmidt(np.random.rand(4,2)@np.random.rand(2,5))
#     print(A2.transpose()@A2)

# no, every nodes are equally important
# Initialize the PageRank vector b0
# Function to perform the power method until convergence with a tolerance level
def power_method_until_convergence(adj_matrix, initial_vector, tolerance=1e-6):
    b_prev = initial_vector
    iterations = 0
    while True:
        # Perform the power method iteration
        b_next = adj_matrix @ b_prev
        
        # Check for convergence by comparing the change to the tolerance
        if np.max(np.abs(b_next - b_prev)) < tolerance:
            break
        
        b_prev = b_next
        iterations += 1
        
    return b_next, iterations

#rank r approximation
U,s,VT = np.linalg.svd(A, full_matrices= False)
Ar = U[:, :r] @ np.diag(s[:r]) @ VT[:r, :]


# Kmeans algorithms 
def dist(x, y):
    """
    this function takes in two 1-d numpy as input an outputs 
    Euclidean the distance between them
    """
    return np.sqrt(np.sum((x - y) ** 2)) ## Fill in the blank: Recall the 'distance' function used in the kMeans algorithm

def kMeans(X, K, maxIters = 20):
    """
    this implementation of k-means takes as input (i) a matrix X
    (with the data points as columns) (ii) an integer K representing the number 
    of clusters, and returns (i) a matrix with the K columns representing 
    the cluster centers and (ii) a list C of the assigned cluster centers
    """
    X_transpose = X.transpose()
    centroids = X_transpose[np.random.choice(X.shape[0], K)]
    for i in range(maxIters):
        # Cluster Assignment step
        C = np.array([np.argmin([dist(x_i, y_k) for y_k in centroids]) for x_i in X_transpose])
        # Update centroids step 
        for k in range(K):
            if (C == k).any():
                centroids[k] = X_transpose[C == k].mean(axis = 0) 
            else: # if there are no data points assigned to this certain centroid
                centroids[k] = X_transpose[np.random.choice(len(X))] 
    return centroids.transpose() , C


'''
write a script to conduct the power iterations for a arbitary symmetric matrix A
with only 1 rank

'''
n = 10 #suppose A in a 10 * 10 matrix
v1 = np.random.rand(n) #randomly generate eigenvalue v1 for decomposition 
v1 = v1 / np.linalg.norm(v1) # normalize v1
#define the n*n symmetric, rank 1 matrix A
A = np.outer(v1, v1)
#define the inital vector b0
b_0 = np.ones(n, dtype = float)/np.sqrt(n)

#power iteration parameters
tolerance = 0.01
iterations = 0
b_k = b_0
error = 1 

#power iterations
while error > tolerance:
    b_k1 = A.dot(b_k)

    #normalize the result
    b_k1_norm = np.linalg.norm(b_k1)
    b_k1 = b_k1 / b_k1_norm

    #calculate the error 
    error = np.linalg.norm(b_k1 - v1)

    #next iteration 
    b_k = b_k1
    iterations += 1

    #break the loop if within the tolerance
    if error <= tolerance:
        break


iterations, error




def gaussian_kernel_matrix(X, sigma):
    """Compute the Gaussian Kernel matrix for the given sigma."""
    n_samples = X.shape[0]
    K = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            K[i, j] = np.exp(-np.linalg.norm(X[i] - X[j])**2 / (2 * sigma**2))
    return K

def k_fold_cross_validation(X, y, sigmas, lambdas, k):
    """Perform k-fold cross validation and return the best sigma and lambda."""
    fold_size = int(len(X) / k)
    best_sigma = None
    best_lambda = None
    best_error = float('inf')

    for sigma in sigmas:
        for lam in lambdas:
            cv_errors = []
            for fold in range(k):
                # Split the dataset into training and validation sets
                start, end = fold * fold_size, (fold + 1) * fold_size
                X_train = np.concatenate([X[:start], X[end:]])
                y_train = np.concatenate([y[:start], y[end:]])
                X_val = X[start:end]
                y_val = y[start:end]

                # Compute the kernel matrices
                K_train = gaussian_kernel_matrix(X_train, sigma)
                K_val = gaussian_kernel_matrix(X_val, sigma)

                # Compute the alpha coefficients
                alpha = np.linalg.inv(K_train + lam * np.eye(len(K_train))) @ y_train

                # Predict the validation set and calculate the error
                predictions = K_val @ alpha
                error = np.mean((predictions - y_val)**2)
                cv_errors.append(error)

            # Compute the average error for the current sigma and lambda
            avg_error = np.mean(cv_errors)
            if avg_error < best_error:
                best_error = avg_error
                best_sigma = sigma
                best_lambda = lam

    return best_sigma, best_lambda, best_error
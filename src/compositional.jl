using Flux

function attention1(X, Wq, Wk, Wv)
    Q = X*Wq
    K = X*Wk
    V = X*Wv
    d = size(K, 2)
    return softmax(Q*K' ./ sqrt(d), dims=2)*V
end

N = 7
hid_dim = 10
dk = 5
dv = 3
head = 2

X = rand(N, hid_dim)
Wq = rand(hid_dim, dk)
Wk = rand(hid_dim, dk)
Wv = rand(hid_dim, dv*head)

function attention2(X, Z, Wq, Wk, Wv)
    Q = X*Wq
    K = Z .* Wk
    V = Z*Wv
    d = size(K, 2)
    return softmax(Q*K' ./ sqrt(d), dims=2)*V
end

dk2 = 11
head2 = 

W̄q = rand(hid_dim, dk2*head2)
W̄k = rand()
W̄v = rand()

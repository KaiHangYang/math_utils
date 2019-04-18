function matrix_to32fv(x) {
    return new Float32Array(x.flat());
}

function matrix_from32fv(x, size) {
    var mat = new Array(size);
    var cnt = 0;
    for (var i = 0; i < size; ++i) {
        mat[i] = new Array(size);
        for (var j = 0; j < size; ++j) {
            mat[i][j] = x[cnt];
            cnt += 1;
        }
    }
    return mat;
}

function matrix_transpose(x) {
    var i,j,m = x.length,n = x[0].length, ret=Array(n),A0,A1,Bj;
    for(j=0;j<n;j++) ret[j] = Array(m);
    for(i=m-1;i>=1;i-=2) {
        A1 = x[i];
        A0 = x[i-1];
        for(j=n-1;j>=1;--j) {
            Bj = ret[j]; Bj[i] = A1[j]; Bj[i-1] = A0[j];
            --j;
            Bj = ret[j]; Bj[i] = A1[j]; Bj[i-1] = A0[j];
        }
        if(j===0) {
            Bj = ret[0]; Bj[i] = A1[0]; Bj[i-1] = A0[0];
        }
    }
    if(i===0) {
        A0 = x[0];
        for(j=n-1;j>=1;--j) {
            ret[j][0] = A0[j];
            --j;
            ret[j][0] = A0[j];
        }
        if(j===0) { ret[0][0] = A0[0]; }
    }
    return ret;
}

function matrix_identity(s) {
    var identity = new Array(s);
    for (var i = 0; i < s; ++i) {
        identity[i] = new Array(s);
        identity[i].fill(0.0)
        identity[i][i] = 1.0
    }

    return identity;
}

function matrix_clone(mat) {
    var m = mat.length, n = mat[0].length;
    var c_mat = Array(m);
    for (var i = 0; i < m; ++i) {
        c_mat[i] = new Array(n);
        for (var j = 0; j < n; ++j) {
            c_mat[i][j] = mat[i][j]
        }
    }

    return c_mat;
}

function matrix_inverse(x) {
    var abs = Math.abs, m = x.length, n = x[0].length;
    var A = matrix_clone(x), Ai, Aj;
    var I = matrix_identity(m), Ii, Ij;
    var i,j,k,x;
    for(j=0;j<n;++j) {
        var i0 = -1;
        var v0 = -1;
        for(i=j;i!==m;++i) { k = abs(A[i][j]); if(k>v0) { i0 = i; v0 = k; } }
        Aj = A[i0]; A[i0] = A[j]; A[j] = Aj;
        Ij = I[i0]; I[i0] = I[j]; I[j] = Ij;
        x = Aj[j];
        for(k=j;k!==n;++k)    Aj[k] /= x; 
        for(k=n-1;k!==-1;--k) Ij[k] /= x;
        for(i=m-1;i!==-1;--i) {
            if(i!==j) {
                Ai = A[i];
                Ii = I[i];
                x = Ai[j];
                for(k=j+1;k!==n;++k)  Ai[k] -= Aj[k]*x;
                for(k=n-1;k>0;--k) { Ii[k] -= Ij[k]*x; --k; Ii[k] -= Ij[k]*x; }
                if(k===0) Ii[0] -= Ij[0]*x;
            }
        }
    }
    return I;
}

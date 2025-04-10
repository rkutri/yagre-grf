def create_data_string(dim, var, ell, nu, nSamp, preString):
    return preString + \
        f"_{dim}d_var{int(100*var)}_ell{int(100*ell)}_nu{int(10*nu)}_{nSamp // 1000}k"

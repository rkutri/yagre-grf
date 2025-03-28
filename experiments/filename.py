def create_data_string(ell, nu, nSamp, preString):
    return preString + f"_ell{int(100*ell)}_nu{int(10*nu)}_{nSamp // 1000}k"

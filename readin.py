import pandas as pd

def df_masks(df):
    mask = (df['psfMagErr_u']<1) & (df['psfMagErr_g']<1) \
        & (df['psfMagErr_r']<1) & (df['psfMagErr_i']<1) \
        & (df['psfMagErr_z']<1)
    df = df[mask]
    mask = (df['psfMag_g']>=18) & (df['psfMag_g']<=22)
    df = df[mask]
    # add colours u-g, g-r, r-i, i-z
    df = df.assign(u_g=(df['psfMag_u']-df['psfMag_g']))
    df = df.assign(g_r=(df['psfMag_g']-df['psfMag_r']))
    df = df.assign(r_i=(df['psfMag_r']-df['psfMag_i']))
    df = df.assign(i_z=(df['psfMag_i']-df['psfMag_z']))
    mask = (df['u_g']<4) & (df['u_g']>-1) & (df['g_r']<4) & (df['g_r']>-1) \
        & (df['r_i']<4) & (df['r_i']>-1) & (df['i_z']<4) & (df['i_z']>-1)
    df = df[mask]
     
    return df
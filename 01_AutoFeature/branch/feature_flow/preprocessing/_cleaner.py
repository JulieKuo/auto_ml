

__all__ = ['cleaner',
           'low',
           'rem',
           'split',
           'clip',
           ]


def cleaner(df):
    cat_df = df.select_dtypes(include=['category', 'object'])
    cat_df = cat_df.apply(low, axis=1).apply(rem, axis=1)
    for name, col in cat_df:
        split(col)


def clip(x):
    # Multiple columns can be clipped at once.
    p01 = x.quantile(0.01)
    p99 = x.quantile(0.99)
    x = x.clip(p01, p99, axis=1)
    return x


def low(x):
    return x.str.lower()


def rem(x):
    return x.str.strip()


def split(x):
    x_split = x.str.split('([A-Za-z]+)', expand=True)
    x_split.columns = [f"{x.name}_{i}" for i in x_split.columns]
    x_split = x_split.loc[:, x_split.nunique() != 1]
    return x_split

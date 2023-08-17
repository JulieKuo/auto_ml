import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
import six


def render_mpl_table(data, col_width=3.0, row_height=0.625, font_size=14,
                     header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
                     bbox=[0, 0, 1, 1], header_columns=0,
                     ax=None, **kwargs):    
    fig = None
    if ax is None:
        size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
        fig, ax = plt.subplots(figsize=size)
        ax.axis('off')

    mpl_table = ax.table(cellText=data.values, cellLoc='center', bbox=bbox, rowLabels=data.index, rowLoc='center', colLabels=data.columns, **kwargs)

    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)
    # mpl_table.auto_set_column_width(col=list(range(len(data.columns))))    # Provide integer list of columns to adjust

    for k, cell in  six.iteritems(mpl_table._cells):
        cell.set_edgecolor(edge_color)
        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
            # cell.get_text().set_rotation(45)
            # cell.set_height(10)
        else:
            cell.set_facecolor(row_colors[k[0]%len(row_colors) ])
    # fig.tight_layout()
    return fig, ax


def output_image(fig, filepath):
    fig.savefig(filepath, bbox_inches='tight', dpi=150)
    return


def by_dfi(df, output_filepath):
    dfi.export(df, output_filepath, table_conversion='matplotlib')
    return


def save_dataframe_as_image(df, filepath):
    fig, ax = render_mpl_table(df, header_columns=0)
    output_image(fig, filepath)
    return

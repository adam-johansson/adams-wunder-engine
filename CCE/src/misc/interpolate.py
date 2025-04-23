from scipy.interpolate import splev, splrep
import pandas as pd
import matplotlib.pyplot as plot
import matplotlib.style
import matplotlib as mpl

mpl.style.use("seaborn-ticks")


def spline(x, y, xnew, k, plt, name):
    func = splrep(x, y, k=k)
    ynew = splev(xnew, func)

    if plt:
        fig, ax = plot.subplots()
        ax.plot(x, y, "s", xnew, ynew, "-", linewidth=1.0, markersize=3)
        ax.legend(["data", "interp k = " + str(k)], loc="best")
        ax.title.set_text(name)
        ax.grid()
        plot.show()

    return ynew


def interp_frame(df, new_index, k, plt, name):
    """Return a new DataFrame with all columns values interpolated
    to the new_index values."""
    df_out = pd.DataFrame(index=new_index)
    df_out.index.name = df.index.name

    for colname, col in df.iteritems():
        df_out[colname] = spline(df.index, col, new_index, k, plt, name + " " + colname)

    return df_out

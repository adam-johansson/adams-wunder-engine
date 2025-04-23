import matplotlib.pyplot as plt

plt.style.use("classic")


def myplot1d(x, y, lab, xname, yname, ax1):
    plt.rcParams.update({"font.size": 15})

    ax1.plot(x, y, label=lab)
    ax1.grid()
    ax1.set_xlabel(xname)
    ax1.set_ylabel(yname)
    plt.legend(loc="best", fontsize="large", frameon=False)


def myplot1d_marker(x, y, lab, xname, yname):
    plt.rcParams.update({"font.size": 15})

    fig, ax1 = plt.subplots()
    ax1.plot(x, y, "s", label=lab)
    ax1.grid()
    ax1.set_xlabel(xname)
    ax1.set_ylabel(yname)
    plt.legend(loc="best", fontsize="large", frameon=False)


def myplot2d(x, y, z, xname, yname, title, zname, lv, map):
    fig, ax = plt.subplots()
    CS = ax.contourf(x, y, z, levels=lv, cmap=map)
    CS2 = ax.contour(CS, colors="gray", linewidths=(1,))
    ax.set_title(title)
    ax.set_xlabel(xname)
    ax.set_ylabel(yname)
    cbar = fig.colorbar(CS, ax=ax)
    cbar.ax.set_ylabel(zname)

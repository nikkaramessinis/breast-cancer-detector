import seaborn as sns

def draw_plots(df_train):
    # Plot 1: Catplot
    h = 2.5
    a = 3
    catplot = sns.catplot(data=df_train, y="cancer", hue="invasive", kind="count", height=h, aspect=a)
    catplot.savefig('catplot.png')

    # Plot 2: Displot
    displot = sns.displot(data=df_train, x="age", hue=df_train[["cancer", "invasive"]].apply(tuple, axis=1), binwidth=1,
                          stat="density", common_norm=False, multiple="stack", height=h, aspect=a)
    displot.savefig('displot.png')

    # Plot 3: Catplot with Box
    boxplot = sns.catplot(data=df_train, y="cancer", x="age", kind="box", orient="h", height=h, aspect=a, width=0.5)
    boxplot.savefig('boxplot.png')
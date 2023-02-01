import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


if __name__ == '__main__':

    data = pd.read_csv('results/Train_uniform-randominit_unet_BCE_adam.csv')

    print(type(data))

    sns.set_theme(style="darkgrid")

    sns.lineplot(x='epoch', y="iou",
                 hue="event",
                 data=data)

    plt.show()
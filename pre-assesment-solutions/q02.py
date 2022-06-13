from pprint import pp
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi

from skimage import color, feature, filters, measure, morphology, segmentation, util
from skimage.data import human_mitosis


if __name__ == "__main__":
    image = human_mitosis()

    thresholds = filters.threshold_multiotsu(image, classes=2)
    regions = np.digitize(image, bins=thresholds)

    cells = image > thresholds[0]
    labeled_cells = measure.label(cells)
    number_of_cells = len(np.unique(labeled_cells))
    print(f"Number of cells: {number_of_cells}")

    # fig, ax = plt.subplots(ncols=2)
    # ax[0].imshow(image)
    # ax[0].set_title("Original")
    # ax[0].axis("off")
    # ax[1].imshow(regions)
    # ax[1].set_title("Segmented Cells")
    # ax[1].axis("off")
    # plt.tight_layout()
    # plt.show()

from dataclasses import dataclass
from pathlib import Path
from datetime import datetime

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.mask import mask as rio_mask
from rasterio.warp import reproject
from rasterio.transform import Affine
from affine import Affine as AffineClass
import geopandas as gpd
from shapely.geometry import mapping, box
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

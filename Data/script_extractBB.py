import rasterio
import geopandas
import numpy as np

precision = 10000

def getBBox(fileName):
    coords = ['minx', 'miny', 'maxx', 'maxy']

    file = open(fileName)
    df = geopandas.read_file(file)
    df_bounds = df.bounds

    final_values = []
    for cc in coords:
        values = []
        for index, row in df.bounds.iterrows():
            values.append(row[cc])
        temp = None
        if 'min' in cc:
            temp = np.amin(values)
        if 'max' in cc:
            temp = np.amax(values)
        final_values.append( temp)
    return final_values

def getGeom(fileName):
    file = open(fileName)
    df = geopandas.read_file(file)
    return df.geometry
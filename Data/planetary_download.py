import shutil
import rasterio.mask
import pystac_client
import planetary_computer
import stackstac
import numpy as np
from script_extractBB import getBBox
import glob
import warnings
import os
import pandas as pd
from datetime import datetime, timedelta


warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)

def createGeoTiff(xarray, outputFileName):
    array_npy = xarray.to_numpy()
    n_bands, height, width = array_npy.shape
    metadata = {
        'driver': 'GTiff',
        'height': height,
        'width': width,
        'count': n_bands,
        'dtype': 'float32',
        'crs': xarray.attrs['crs'].upper(),
        'transform': xarray.attrs['transform']
    }
    with rasterio.open(outputFileName, 'w', **metadata) as dst:
        dst.write(array_npy)


def getRasterData(extent, prefix, base_geojson_path, base_out_path, start_date, end_date):
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )

    collections = ['sentinel-2-l2a']
    time_range = f'{start_date}/{end_date}'
    search = catalog.search(
        collections=collections,
        datetime=time_range,
        query={},
        bbox=extent
    )

    items = search.get_all_items()

    os.makedirs(os.path.join(base_out_path, 'sentinel_2'), exist_ok=True)
    os.makedirs(os.path.join(base_geojson_path, 'downloaded'), exist_ok=True)
    best_cloud_cover_item = {}
    for item in items:
        bands = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12', 'SCL']
        FILL_VALUE = 0
        proj = item.properties.get('proj:epsg')
        array = stackstac.stack(
            item,
            assets=bands,
            resolution=10,
            dtype="uint16",
            fill_value=FILL_VALUE,
            epsg=proj,
            bounds_latlon=extent)
        label_dict = {0.: 'NO DATA', 1.: 'SATURED OR DEFECTIVE', 2: 'DARK AREA PIXELS', 3.: 'CLOUD SHADOWS',
                      4.: 'VEGETATION', 5.: 'NOT VEGETATED',
                      6.: 'WATER', 7.: 'UNCLASSIFIED', 8.: 'CLOUD MEDIUM PROB', 9.: 'CLOUD HIGH PROB',
                      10.: 'THIN CIRRUS', 11.: 'SNOW'}
        array = array.isel(time=0)

        scl_band = array.sel(band='SCL').values
        cloud_mask = np.where((scl_band == 3) | (scl_band == 8) | (scl_band == 9) | (scl_band == 10), 1, 0)
        item_metrics = dict()
        item_metrics['cloud_coverage'] = (cloud_mask.sum() / cloud_mask.size) * 100

        cloud_mask_w_no_data = np.where(
            (scl_band == 3) | (scl_band == 8) | (scl_band == 9) | (scl_band == 10) | (scl_band == 0) | (
                        scl_band == 1) | (scl_band == 2), 1, 0)
        item_metrics['cloud_coverage_w_no_data'] = (cloud_mask_w_no_data.sum() / cloud_mask_w_no_data.size) * 100

        eval_metric = 'cloud_coverage_w_no_data'
        if (best_cloud_cover_item == {} or (item_metrics[eval_metric] < best_cloud_cover_item[eval_metric]) or
                (item_metrics[eval_metric] <= best_cloud_cover_item[eval_metric] and item.properties['datetime'] >
                 best_cloud_cover_item['datetime'])):
            best_cloud_cover_item['datetime'] = item.properties['datetime']
            best_cloud_cover_item['mgrs'] = item.properties['s2:mgrs_tile']
            best_cloud_cover_item['cloud_coverage'] = item_metrics['cloud_coverage']
            best_cloud_cover_item['cloud_coverage_w_no_data'] = item_metrics['cloud_coverage_w_no_data']
            best_cloud_cover_item['data'] = array
            best_cloud_cover_item['id'] = item.id
            best_cloud_cover_item['s2_epsg'] = item.properties.get('proj:epsg')
            best_cloud_cover_item['num_json'] = prefix

    if best_cloud_cover_item == {}:
        print('[ATTENZIONE] Nessuna Ground Truth senza nuvole è stata trovata!')
    else:
        outputFileName = os.path.join(base_out_path, 'sentinel_2', f'geojson_{prefix}.tif')
        createGeoTiff(best_cloud_cover_item['data'], outputFileName)
        del best_cloud_cover_item['data']
        s_id = best_cloud_cover_item['id']
        proj = best_cloud_cover_item['s2_epsg']
        print(f'====S2 ({proj}) {s_id}====')
        get_sentinel_1_data(extent=extent, s2_attr=best_cloud_cover_item, base_out_path=base_out_path)
        geojson_filename = f'geojson_{prefix}.geojson'
        shutil.move(os.path.join(base_geojson_path, geojson_filename), os.path.join(base_geojson_path, 'downloaded', geojson_filename))
    return best_cloud_cover_item


def get_sentinel_1_data(extent, s2_attr, base_out_path):
    s2_date = s2_attr['datetime']
    s2_proj = s2_attr['s2_epsg']
    id = s2_attr['num_json']
    dt_object = datetime.strptime(s2_date, '%Y-%m-%dT%H:%M:%S.%fZ')
    # Extract the S2 date
    formatted_date = dt_object.date().strftime('%Y-%m-%d')
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    collections = ['sentinel-1-rtc']
    s1_bands = ["VV", "VH"]
    try:
        search = catalog.search(
            collections=collections,
            query={
                'sar:polarizations': {"eq": s1_bands},
                'proj:epsg': {"eq": s2_proj}
            },
            datetime=formatted_date,
            bbox=extent
        )
        items = search.get_all_items()
        os.makedirs(f'{base_out_path}/sentinel_1/', exist_ok=True)
        if len(items) == 0:
            date_interval = 0
            while len(items) == 0:
                date_interval += 1
                date_before = (dt_object - timedelta(date_interval)).strftime('%Y-%m-%d')
                date_after = (dt_object + timedelta(date_interval)).strftime('%Y-%m-%d')
                str_date_interval = f'{date_before}/{date_after}'
                search = catalog.search(
                    collections=collections,
                    query={
                        "sar:polarizations": {"eq": s1_bands},
                        'proj:epsg': {"eq": s2_proj}
                    },
                    datetime=str_date_interval,
                    bbox=extent
                )
                print(f'Allargando date {str_date_interval}')
                items = search.get_all_items()
            print(f'Trovate {str(len(items))} rilevazioni Sentinel 1 per geojson_{id} in date {str_date_interval}')
        item = items[0]
        proj = item.properties.get('proj:epsg')
        print(f's1 ({proj}) => {item.id}')
        FILL_VALUE = 0
        array = stackstac.stack(planetary_computer.sign(item),
                                bounds_latlon=extent,
                                assets=['vv', 'vh'],
                                fill_value=FILL_VALUE,
                                epsg=proj,
                                resolution=10)
        outputFileName = f'{base_out_path}/sentinel_1/geojson_{id}.tif'
        data = array.isel(time=0)
        createGeoTiff(data, outputFileName)
    except Exception as e:
        print(f'Error downloading Sentinel 1: {e}')


def main():
    base_out_path = "france_october_planetary"
    base_geojson_path = 'geoJson'
    start_date = '2018-10-01'
    end_date = '2018-10-31'

    if not os.path.exists(base_out_path):
        os.makedirs(base_out_path)

    filenames = glob.glob(os.path.join(base_geojson_path,'*.geojson'))
    best_results = []
    try:
        for filename in filenames:
            print("processing %s" % filename)
            # notes that every geojson have to be expressed as geojson_<id>.geojson
            # it's MANDATORY to have numerical id in filename
            prefix = int(os.path.basename(filename).split(".")[0].split("_")[1])
            extent = getBBox(filename)
            best_cloud_cover_item = getRasterData(extent=extent, prefix=prefix, base_geojson_path=base_geojson_path,
                                                  base_out_path=base_out_path, start_date=start_date, end_date=end_date)
            best_results.append(best_cloud_cover_item)
    except Exception as e:
        print(f'Errore {e}..... salvataggio csv...')

    df_res = pd.DataFrame(best_results)
    df_res.to_csv('planetary_results.csv', sep="\t", index=False)

if __name__ == "__main__":
    main()
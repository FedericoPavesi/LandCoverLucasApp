import streamlit as st
from PIL import Image
from geemap import foliumap
import ee



st.markdown('# Database creation')

path = 'C:/Users/drikb/Desktop/Tirocinio/Presentation_app/'

### LUCAS PRESENTATION ########################################################

st.markdown('### Eurostat Lucas')

st.markdown('Eurostat Lucas (Land Use and Cover Area frame Survey) points consists in a __collection of geo-referenced points__ (in WGS84 geodetic system) with an __associated land cover and land use three-digits classification__. Alongside with this information we also find the __date of the survey for each point__ (which will be extremely useful in later steps). The survey takes place every 3 years since 2006 resulting in five collections respectively referred to 2006, 2009, 2012, 2015 and 2018. Unfortunately, Covid-19 pandemic of 2020 delayed 2021 survey to 2022, and it will end in September of the same year. We shall take advantage of information contained __only in 2018 survey__ because, as we will better discuss when analysing remote sensing data source, complementary information from satellites detections is available only starting from 2017. The survey for 2018 cover 27 Europe countries, for a __total amount of 237,768 points__ (considering we have removed points which encountered recording issues). Each point is classified in a three-digits land cover system, which __first digit split land cover in eight classes__: Artificial Land (A), Cropland (B), Woodland (C), Shrubland (D), Grassland (E), Bareland (F), Water (G) and Wetlands (H). Every main category is then divided into a two-level classification of subclasses. The classification for the __first digit is suitable for training our classifiers__, as we are not interested in having more details about ground composition. At the end what we have is a set of geo-referenced points with an 8 classes classification of ground truth. ')

lucas_numerosity = Image.open(path + 'Lucas_country_numerosity.png')
lucas_LC1 = Image.open(path + 'Lucas_LC1_numerosity.png')

st.image(lucas_numerosity,
         caption = 'Per Country Lucas points 2018 numerosity')

st.image(lucas_LC1,
         caption = 'Per one-digit class Lucas points 2018 numerosity')


## SENTINEL - 2 PRESENTATION ##################################################

st.markdown('### Sentinel-2')

st.markdown('For the aim of our analysis, __Copernicus Sentinel-2 satellites__ are the most suitable as they specifically monitor __land surface__. Sentinel-2 mission makes use of two polar-orbiting satellites belonging to the same sun-synchronous area, with a phase of 180° to each other. This mission provides wide-swath , __high-resolution__, __multi-spectral imaging data__ with a revisit time of approximately five days. Images are multi-spectral as satellites carry an __MSI (Multi Spectral Instrument) optical instrument__, which samples __thirteen spectral bands__. Images are high resolution as four out of the thirteen bands have a 10 meters resolution (red, green, blue and near-infrared) while remaining nine are divided in six with 20 meters and three with 60 meters resolution. It is defined wide-swath as the orbital swath width is 290km.  Rasters  containing spectral bands reflectance values will be the main source of surface reflectance data for this work. Of course, raw MSI response is not open access but, as we are not interested into dealing with unprocessed data, which anyway must be handled properly, we are fine with the available products: in particular we opted for level-2A products, which register bottom-of-atmosphere reflectance , in cartographic geometry. This product is available since 2017 ; it could be possible to perform atmospheric correction from level-1C data (collecting top of atmosphere reflectance) of which we have older detections but, as it would overshoot the objectives of this research, we will stick to what is provided already properly processed.')


## IDEA PRESENTATION ##########################################################

st.markdown('### The process')

# TEXT ----------------------------------------------------

st.markdown('The merging process between Lucas points and sentinel-2 images is performed using earth engine library. The information we need from Lucas comprehends its land cover class, latitude and longitude in world geodetic system 1984 (WGS84) and the date when it was registered. We use all the Lucas points available for 2018, which cover most of the euro-area countries.  A random sample, balanced over the one-digit categories is then taken with 5,000 elements for each one-digit class, for a total of 40,000 points (out of around 237,000). For two categories, water and wetlands, the whole dataset does not contain enough points to reach the objective. To overcome this obstacle, we performed a simple step of data augmentation consisting in artificially modify the survey date of some points, shifting them by six months in the future or in the past. The same point will be downloaded in different periods of time so that, under the assumption there were no major bio changes in the soil condition, we will add the heterogeneity we are looking for to build a proper database. We now have all the information to launch the query on earth engine, we set the maximum cloud cover percentage equal to 60%. We can choose the shape in pixels of images we are going to download (we selected 3x3, so that we can obtain both 1x1 and 3x3 from a single download). It is also possible to select which spectral bands we are interested into downloading: we select all level-2A bands. From previous chapter, we know not all spectral bands have the same resolution, but there are higher and lower resolution bands. Because we are interested into 10 meters resolution, we must resample lower resolution bands. To do that, the safest way consists in just splitting the lower resolution pixel into smaller (higher resolution) pixels. For example, a pixel of a band with 20 meters resolution and reflectance value ∝ would be split into four 10 meters pixels, each with value ∝. For each point, the loop looks for all rasters are available in a time span of two month before and one after the survey date. We now have a collection of images; recall given the same projection pixels should overlap, we can employ a reducer method to reduce the collection to an image composed by the median reflectance of each pixel. Of course, it is possible to have some raster covering the point in different projections, fortunately Earth Engine reducer method is projection insensitive, allowing us to trust the approximation of pixels’ position is optimal. Having this image of median reflectance, we extract the area of interest (the point and its neighbourhood) using an Earth Engine reduce region method; specifying the projection and the scale in this routine allows to resample lower resolution bands as we explained before. Hence, for each point we get a 3x3 pixels image (covering a 30x30 meters of earth surface) with an associated land cover lass. The collection of all sampled points constitutes a dataset with one column for land cover class and one for the relative image, with 40,000 rows. Through some code optimization, we get the download time to be only around 3 to 4 hours. This is a relatively low computing time considering how many information we are processing in this routine. Moreover, the process is not computing power intensive as it does not require to perform complex calculus locally thanks to the possibility to employ almost only the power of earth engine, meaning we just need a good internet connection. The dataset obtained is not yet ready for our algorithms: because we are going to employ random forest and multi-layer perceptron, an image cannot be fed directly as we downloaded it. We need to flatten all the images to obtain a database with a ‘classic shape’, scilicet to obtain each row corresponding to one point, the first column being the land cover class and all the others contain reflectance of spectral bands of each image’s pixel. To make an example, an image of shape 3x3 with 12 spectral bands will become a row with 109 columns: 108 for spectral bands reflectance of each pixel and 1 for the land cover class.')

# CODE --------------------------------------------------

start_date = '2018-01-01'
end_date = '2018-12-31'
cloud_filter = 20
bands = ['TCI_R', 'TCI_G', 'TCI_B']

s2a = (ee.ImageCollection('COPERNICUS/S2_SR').
       filterDate(start_date, end_date)
       .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', cloud_filter)))


mosaic = s2a.select(bands).reduce(ee.Reducer.median())

mosaic = mosaic.uint8()


lucas = ee.FeatureCollection('projects/sentinel2download332224/assets/lucas_full')

Map = foliumap.Map(center = [41.902782, 12.496366],
                   zoom = 8)

vizParams = {
  'bands': ['TCI_R_median', 'TCI_G_median', 'TCI_B_median'],
  'min': 0,
  'max': 255
}

lucas_par = ee.Dictionary({'A' : '#ff0101',
             'B' : '#ffff01',
             'C' : '#336601',
             'D' : '#ff8001',
             'E' : '#01ff01',
             'F' : '#808080',
             'G' : '#0101ff',
             'H' : '#99ffff'})


color_dict = {'Artificial Land' : '#ff0101',
             'Cropland' : '#ffff01',
             'Woodland' : '#336601',
             'Shrubland' : '#ff8001',
             'Greenland' : '#01ff01',
             'Bareland' : '#808080',
             'Water' : '#0101ff',
             'Wetlands' : '#99ffff'}


def pointstyle(f):
    kl = f.get('LC1')
    return f.set({'style' : {'fillColor' : lucas_par.get(kl)}})

styled = lucas.map(pointstyle).style(styleProperty = 'style')


Map.addLayer(mosaic, vizParams, 'Sentinel-2 2018 median')
Map.addLayer(styled, name =  'Lucas Points')
Map.add_legend(legend_title="NLCD Land Cover Classification", legend_dict=color_dict)

Map.to_streamlit()








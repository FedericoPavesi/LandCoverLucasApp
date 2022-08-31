import streamlit as st
from geemap import foliumap
from PIL import Image
import ee

st.markdown('# Map classification')

st.markdown('In previous section, we evaluated performances in a train, validation and test approach. This is very good for a first evaluation of the model but, as we would like to assess the real-case performance of a land-cover classifier, we cannot rely only on test accuracy and confusion matrix. Ideally, we would like to perform a comparison between a reliable ground truth classification mask on sentinel-2 images and our classification. Such information is unfortunately not available but we have an alternative: Lucas statistics. We know this dataset contains land cover areal statistics grouped by NUTS until level 2. What we can do is to create an image covering only a certain area (in this case we will use a region), then apply the classifier on the whole region to compute land cover statistics. Finally, we compare results from the classifier with ground truth coming from Lucas statistics. The intuition is again very simple: the closest we get to Lucas regional statistics, the better the classifier in predicting land cover classes.\n First step is the creation of reflectance map of the region of interest. Because this process is quite computational expensive, we shall choose a single region and focus on it. This limitation weakens the information of this evaluation procedure, through which we could have enlighten strengths and weaknesses of a classifier. Moreover, this exposes us to the risk of underestimating or overestimating the accuracy of the map. Suppose for example the case of a region with each land cover class almost perfectly balanced (for 8 classes, each one covers 12.5% of region surface), in this case we would find a random classifier to produce incredibly high performances both in terms of accuracy and computational speed. Anyway, this evaluation would be not correct as the prediction for each pixel is purely casual. This risk is of course reduced by the train, validation and test assessment, but still the lack of numerous samples (regions) jeopardizes the validity of any proposition based on these results. Evaluation would consequentially refer mainly to what we discussed in previous paragraph; here we are just showing a single unit of a wider procedure which helps evaluating land cover classifiers when performing their ultimate objective. For this example, we chose Italy region Lazio. We again employ Earth Engine. We start by defining the region of interest as a multi-polygon representing ISTAT administrative boundaries of the region. We launch a query to retrieve all the images covering this area in 2018 with cloud coverage lower than 40%. We then reduce the image collection to median reflectance of each pixel (method is the same as for the creation of the database explained in previous section). At this point, we use the shape of the multi-polygon to crop the image, this way we obtain a map in which only pixels belonging to Lazio territory would be saved while out-of-area pixels are excluded. Finally, we export the image to Google Drive  divided in lower size slices and we download locally each of them. Because we need to have squared slices, out-of-area points are recorded with each spectral band value to be equal to zero. At this point we apply RF and MLP on all the slices and compute regional statistics on the predictions. To apply 3x3 classifiers, we use a striding procedure in which for each single pixel it takes into account its 3x3 neighbourhood. Moreover, for 3x3 algorithms it was necessary to add an artificial margin in order to avoid border issues. Specifically, the padding procedure uses the edge values of the image.')

st.markdown('Below, it is possible to appreaciate land cover masks for each classification algorithm (they must be activated in the upper-right hand side of the map) on region Lazio')


Map = foliumap.Map(center = [41.902782, 12.496366],
                   zoom = 8)


lazio_TCI = ee.Image('users/federicopavesiwork/Lazio_2018_TCI')
mlp11 = ee.Image('users/federicopavesiwork/MLP_1x1')
mlp33 = ee.Image('users/federicopavesiwork/MLP_3x3')
rf11 = ee.Image('users/federicopavesiwork/RF_1x1')
rf33 = ee.Image('users/federicopavesiwork/RF_3x3')

viz_params = {'min' : 0, 
              'max' : 8,
             'palette' : ['#000000', '#ff0101', '#ffff01', '#336601', '#ff8001',
                         '#01ff01', '#808080', '#0101ff', '#99ffff']}

rgb_params = {'min' : 1, 'max' : 255}

color_dict = {'Artificial Land' : '#ff0101',
             'Cropland' : '#ffff01',
             'Woodland' : '#336601',
             'Shrubland' : '#ff8001',
             'Greenland' : '#01ff01',
             'Bareland' : '#808080',
             'Water' : '#0101ff',
             'Wetlands' : '#99ffff'}

Map.addLayer(lazio_TCI, rgb_params, name = 'Lazio TCI')
Map.addLayer(mlp11, viz_params, name = 'MLP 1x1', shown = False)
Map.addLayer(mlp33, viz_params, name = 'MLP 3x3', shown = False)
Map.addLayer(rf11, viz_params, name = 'RF 1x1', shown = False)
Map.addLayer(rf33, viz_params, name = 'RF 3x3', shown = False)
Map.add_legend(legend_title="NLCD Land Cover Classification", legend_dict=color_dict)


Map.to_streamlit()


st.markdown('Figures below provide an overview of the comparison between Lucas statistics and statistics obtained by applying our classifiers (respectively, random forest 1x1, multi-layer perceptron 1x1, random forest 3x3 and multi-layer perceptron 3x3). Notice for RF 3x3 we were only able to compute two trials as computational time was prohibitive. What we can appreciate is all algorithms produce distributions relatively close to each other, with a Kullback-Leibler divergence from Lucas distribution of around 0.42. RF 3x3 seems the closest one with a score of around 0.41, while conversely MLP 3x3 seems most different scoring 0.46. Unfortunately, even considering the closest prediction, we are far from producing a reliable soil classification. It’s easy to notice from figure 3x3 all classifiers tend to largely overestimate shrubland, wetlands and water; while at the same time they considerably underestimate croplands and woodlands. Anyway, is something we should expect: as mentioned previously, land cover classes are defined by something that goes beyond pure reflectance values, they are complex structures of somehow interacting pixels. Sticking to previous example, a cropland could be defined as a specific structure of a network of pixels: if we see only one green pixel, it is hard to tell what it might belong to, looking instead at a parallelepiped shaped cluster of green pixels, we are prone to think it is a cropland, if we see a collection of adjacent structures of this kind (with eventually some interruption as a street for example) confidence our guess was correct is significantly increased. Moreover, we can spot two other sources of bias in this approach. First, as we already discussed, choosing pixel’s median reflectance especially over one year involves seasonality issues (like snow coverage , water level or croplands stages). Second, it is not simple to tell in which measure Lucas statistics are precise as they are computed using a survey approach (manually weighting recorded points) and we might assume they had to face issues similar to what we encountered (as for example seasonality for water levels).')

path = 'C:/Users/drikb/Desktop/Tirocinio/Presentation_app/'

mlp1x1 = Image.open(path + 'Map_bar_MLP1x1.png')
mlp3x3 = Image.open(path + 'Map_bar_MLP3x3.png')
rf1x1 = Image.open(path + 'Map_bar_RF1x1.png')
rf3x3 = Image.open(path + 'Map_bar_RF3x3.png')

st.image(rf1x1, caption = 'RF 1x1 results compared to Lucas statistics')
st.image(mlp1x1, caption = 'MLP 1x1 results compared to Lucas statistics')
st.image(rf3x3, caption = 'RF 3x3 results compared to Lucas statistics')
st.image(mlp3x3, caption = 'MLP 3x3 results compared to Lucas statistics')

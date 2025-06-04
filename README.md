# HackingLimno2025
Notebooks and code for the Hacking Limno 2025 Workshop on biodiversity data.

[abstract](https://docs.google.com/document/d/1MUzUd9bXD4eNiDiEqdbZ7VlchHqBCh-QKscqf4aqxTg/edit?tab=t.0)

## steps
1. find relevant OBIS & ERDDAP data 
2. build OBIS samples table
3. add environmental data to samples table from ERDDAP
4. use samples table to characterize taxa
    * what environments does the taxa live in
    * SDM model 

## references

Notebooks in progress:
* [tylar pyobis-corals based](https://colab.research.google.com/drive/1L8XN3KKgfwC-32axxv3tspX0-e2CQhGd?usp=sharing)
* TODO: erdappy query
* erdap + obis sampling to create occurence+environmental table
* samples into SDM

data sources
* [list of EOV aphiaIDs](https://github.com/ioos/marine_life_data_network/blob/main/eov_taxonomy/IdentifierList.csv)
* seascapes (lots of environmental information in one product)

SDM examples & frameworks
Ideas:

example       | env. data sources                              | framework           | ref
------------- | ---------------------------------------------- | ------------------- | ---------
daniel furman | worldClim                                      | scikitlearn         | [ref](https://daniel-furman.github.io/Python-species-distribution-modeling/)
Google        | worldClim, SRTM elevation, global forest cover | Google Earth Engine | [ref](https://developers.google.com/earth-engine/tutorials/community/species-distribution-modeling/species-distribution-modeling)
elapid        | NA                                             | elapid              | [ref](https://github.com/earth-chris/elapid)
 

pyobis examples
* [query & display corals](https://github.com/iobis/pyobis/blob/main/notebooks/biodiversity_mapping.ipynb)
* [query for specific dataset](https://ioos.github.io/ioos_code_lab/content/code_gallery/data_access_notebooks/2022-11-23_pyobis_example.html)

other obis examples
* [robis EOV H3 grids](https://github.com/NOAA-GIS4Ocean/BioEco_EOV/blob/main/EOV_obisindicators_hex.R)
* [robis EOV queries](https://ioos.github.io/ioos_code_lab/content/code_gallery/data_analysis_and_visualization_notebooks/2024-09-13-OBIS_EOVs.html)

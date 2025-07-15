# HackingLimno2025
Notebooks and code for the [Hacking Limno 2025 Workshop](https://aquaticdatasciopensci.github.io/) on biodiversity data.

[Accompanying presentation](https://docs.google.com/presentation/d/1BDsl73n5h1Ka38kTdU8vA1XCmb7olyJjstp_1SBIOp0/edit?usp=sharing)

The Global Ocean Observing System (GOOS), with NOAA's U.S. Integrated Ocean Observing System (IOOS) as part of it, uses Biological and Ecological Essential Ocean Variables (BioEco EOVs) to standardize ocean observing data from communities like the Marine Biodiversity Observation Network (MBON). The GOOS Biology and BioEco Variables focus on the abundance and distribution of key aquatic organisms. Using predefined lists of species, one can query biological occurrence data from the Ocean Biodiversity Information System (OBIS) and the Global Biodiversity Information Facility (GBIF). After querying, users can analyze OBIS and GBIF occurrence data to study the abundance and distribution of these BioEco Variables. This occurrence data can then be combined with gridded and tabular environmental data served by ERDDAP to further analyze into products (e.g. species distribution models). In this workshop we will demonstrate the tools and techniques for assessing ecosystem health using this open science framework.


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


## Session title: Mapping biodiversity indicator species using open data

## Lead(s): 
Tylar Murray (tylar.murray@usf.edu, murray.tylar@gmail.com), Mathew Biddle (mathew.biddle@noaa.gov) 

## Date: 
July 21-23

## Duration: 
1.5hrs

## Abstract:
The Global Ocean Observing System (GOOS), with NOAA's U.S. Integrated Ocean Observing System (IOOS) as part of it, uses Biological and Ecological Essential Ocean Variables (BioEco EOVs) to standardize ocean observing data from communities like the Marine Biodiversity Observation Network (MBON). The GOOS Biology and BioEco Variables focus on the abundance and distribution of key aquatic organisms. Using predefined lists of species, one can query biological occurrence data from the Ocean Biodiversity Information System (OBIS) and the Global Biodiversity Information Facility (GBIF). After querying, users can analyze OBIS and GBIF occurrence data to study the abundance and distribution of these BioEco Variables. This occurrence data can then be combined with gridded and tabular environmental data served by ERDDAP to further analyze into products (e.g. species distribution models). In this workshop we will demonstrate the tools and techniques for assessing ecosystem health using this open science framework. 

## Resources:
* Robis query to OBIS for EOVs: https://ioos.github.io/ioos_code_lab/content/code_gallery/data_analysis_and_visualization_notebooks/2024-09-13-OBIS_EOVs.html 
* Using pyobis - https://ioos.github.io/ioos_code_lab/content/code_gallery/data_access_notebooks/2022-11-23_pyobis_example.html 
* Searching for data across multiple ERDDAPs: https://ioos.github.io/ioos_code_lab/content/code_gallery/data_access_notebooks/2021-10-19-multiple-erddap-search.html 
